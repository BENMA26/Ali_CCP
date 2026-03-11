import os
import time
import torch
import numpy as np
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.model import TwoTowerModel,RecallLoss
from src.dataset import AliCCPDataset,AliCCPPairDataset
import torch.nn.functional as F

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_single(batch, model):
    """
    Single Loss

    batch : (user_feature, item_feature, label)
        user_feature  : (B, user_input_dim)
        item_feature  : (B, item_input_dim)
        label         : (B,)  0 / 1

    思路：
        计算 user 与 item 的内积得分，用 BCE Loss 做二分类。
    """
    user_feature, item_feature, label = batch

    user_emb, item_emb = model(user_feature, item_feature)  # (B, D), (B, D)

    # 逐元素点积 → 标量得分  (B,)
    score = (user_emb * item_emb).sum(dim=-1)

    label = label.float()
    loss  = F.binary_cross_entropy_with_logits(score, label)

    return loss

def train_pair(batch, model):
    """
    Pair Loss

    batch : (user_feature, pos_item_feature, neg_item_feature)
        user_feature        : (B, user_input_dim)   —— 同一批用户
        pos_item_feature    : (B, item_input_dim)   —— 对应正样本
        neg_item_feature    : (B, item_input_dim)   —— 对应负样本（同一用户未点击）

    思路：
        优化 pos_score > neg_score, 即最大化 sigmoid(pos_score - neg_score)。
    """
    user_feature, pos_item_feature, neg_item_feature = batch

    # 正负样本使用同一个 user embedding
    user_embedding     = model.get_user_embedding(user_feature)       # (B, D)
    pos_item_embedding = model.get_item_embedding(pos_item_feature)   # (B, D)
    neg_item_embedding = model.get_item_embedding(neg_item_feature)   # (B, D)

    pos_score = (user_embedding * pos_item_embedding).sum(dim=-1)     # (B,)
    neg_score = (user_embedding * neg_item_embedding).sum(dim=-1)     # (B,)

    # BPR Loss：-log σ(pos_score - neg_score)
    loss = - F.logsigmoid(pos_score - neg_score).mean()

    return loss

def train_info_nce(batch, model, temperature=0.05):
    """
    InfoNCE Loss（对比学习）。

    batch : (user_feature, item_feature)
        user_feature  : (B, user_input_dim)   —— 每行是一个正样本对的用户
        item_feature  : (B, item_input_dim)   —— 每行是对应的正样本物品

    思路：
        构建 (B, B) 相似度矩阵，对角线是正样本得分，
        其余位置是 batch 内其他用户的正样本物品（即负样本）。
        用交叉熵让模型在 B 个候选中识别出正样本（对角线位置）。

    假负样本处理：
        若 batch 内某个负样本与当前用户的相似度超过阈值 δ，
        则将其 mask 掉，不参与 loss 计算。
    """
    user_feature, item_feature = batch

    user_emb, item_emb = model(user_feature, item_feature)  # (B, D), (B, D)

    B = user_emb.size(0)

    # (B, B) 相似度矩阵，除以温度系数
    logits = (user_emb @ item_emb.T) / temperature          # (B, B)

    # ---- 假负样本 mask ----
    # 相似度超过阈值的位置视为潜在正样本，设为极小值使其不影响 softmax
    false_negative_mask = logits.detach() > 0.8             # (B, B)
    # 对角线（真正样本）不能被 mask 掉
    eye_mask = torch.eye(B, dtype=torch.bool, device=logits.device)
    false_negative_mask = false_negative_mask & ~eye_mask
    logits = logits.masked_fill(false_negative_mask, -1e9)

    # 标签：对角线位置为正样本
    labels = torch.arange(B, device=logits.device)          # (B,)

    # 对称 loss：用户侧 + 物品侧取平均，让两个塔都得到充分训练
    loss_u2i = F.cross_entropy(logits,   labels)            # 每行：user 找 item
    loss_i2u = F.cross_entropy(logits.T, labels)            # 每列：item 找 user
    loss     = (loss_u2i + loss_i2u) / 2

    return loss

def train_loop(
    model,
    train_dataset,
    val_dataset,
    mode: str = "list",           # "single" | "pair" | "list"
    epochs: int         = 20,
    batch_size: int     = 2048,
    lr: float           = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float  = 0.05,   # InfoNCE temperature
    num_workers: int    = 4,
    device_str: str     = "auto",
    save_dir: str       = "checkpoints",
    log_dir: str        = "runs",
    save_every: int     = 1,
    eval_every: int     = 1,
    early_stop: int     = 5,
    resume_path: str    = None,
    seed: int           = 42,
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Device] {device}  |  Mode: {mode}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.to(device)

    if mode == "single":
        # Pointwise BCE
        criterion = nn.BCEWithLogitsLoss()

    elif mode == "pair":
        def bpr_loss(pos_score, neg_score):
            # pos_score: (B,), neg_score: (B, 1)
            return -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-8).mean()
        criterion = bpr_loss

    else:
        # Listwise InfoNCE
        criterion = RecallLoss(mode="infonce", temperature=temperature)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{mode}_{int(time.time())}"))

    start_epoch = 0

    best_metric   = -float("inf")   # 以 HR@50 为主指标
    patience_cnt  = 0
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(start_epoch, epochs):
        '''
        training stage
        '''
        model.train()
        total_loss    = 0.0
        total_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in pbar:
            # Move batch to device
            batch = tuple(x.to(device) for x in batch)

            optimizer.zero_grad()

            if mode == "single":
                loss = train_single(batch, model)
            elif mode == "pair":
                loss = train_pair(batch, model)
            else:
                loss = train_info_nce(batch, model, temperature)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            total_loss    += loss.item()
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / total_batches
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1:>3}/{epochs}]  "
              f"train_loss={avg_train_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("LR",         scheduler.get_last_lr()[0], epoch)
        '''
        validation stage
        '''
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            # 用于统计正负样本得分
            pos_scores_list = []
            neg_scores_list = []

            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                for batch in pbar_val:
                    # Move batch to device
                    batch = tuple(x.to(device) for x in batch)

                    if mode == "single":
                        loss = train_single(batch, model)
                        # 计算得分并按 label 分类
                        user_feature, item_feature, label = batch
                        user_emb, item_emb = model(user_feature, item_feature)
                        scores = (user_emb * item_emb).sum(dim=-1)

                        pos_mask = label == 1
                        neg_mask = label == 0
                        if pos_mask.any():
                            pos_scores_list.append(scores[pos_mask].cpu())
                        if neg_mask.any():
                            neg_scores_list.append(scores[neg_mask].cpu())

                    elif mode == "pair":
                        loss = train_pair(batch, model)
                        # 计算正负样本得分
                        user_feature, pos_item_feature, neg_item_feature = batch
                        user_embedding = model.get_user_embedding(user_feature)
                        pos_item_embedding = model.get_item_embedding(pos_item_feature)
                        neg_item_embedding = model.get_item_embedding(neg_item_feature)

                        pos_scores = (user_embedding * pos_item_embedding).sum(dim=-1)
                        neg_scores = (user_embedding * neg_item_embedding).sum(dim=-1)

                        pos_scores_list.append(pos_scores.cpu())
                        neg_scores_list.append(neg_scores.cpu())

                    else:
                        loss = train_info_nce(batch, model, temperature)
                        # InfoNCE 模式：对角线是正样本，其余是负样本
                        user_feature, item_feature = batch
                        user_emb, item_emb = model(user_feature, item_feature)

                        B = user_emb.size(0)
                        logits = (user_emb @ item_emb.T) / temperature

                        # 对角线是正样本得分
                        pos_scores = torch.diagonal(logits)
                        pos_scores_list.append(pos_scores.cpu())

                        # 非对角线是负样本得分
                        mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
                        neg_scores = logits[mask]
                        neg_scores_list.append(neg_scores.cpu())

                    val_loss += loss.item()
                    val_batches += 1
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")

            avg_val_loss = val_loss / val_batches

            # 计算正负样本平均得分
            avg_pos_score = torch.cat(pos_scores_list).mean().item() if pos_scores_list else 0.0
            avg_neg_score = torch.cat(neg_scores_list).mean().item() if neg_scores_list else 0.0
            score_gap = avg_pos_score - avg_neg_score

            print(f"         val_loss={avg_val_loss:.4f}  "
                  f"pos_score={avg_pos_score:.4f}  "
                  f"neg_score={avg_neg_score:.4f}  "
                  f"gap={score_gap:.4f}")

            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Score/pos", avg_pos_score, epoch)
            writer.add_scalar("Score/neg", avg_neg_score, epoch)
            writer.add_scalar("Score/gap", score_gap, epoch)

            # Early stopping based on validation loss (lower is better)
            current_metric = -avg_val_loss
            if current_metric > best_metric:
                best_metric = current_metric
                patience_cnt = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                }, best_ckpt_path)
                print(f"         → Best model saved (val_loss={avg_val_loss:.4f})")
            else:
                patience_cnt += 1
                print(f"         → No improvement ({patience_cnt}/{early_stop})")
                if patience_cnt >= early_stop:
                    print(f"\n[Early Stop] No improvement for {early_stop} epochs.")
                    break

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)
            print(f"         → Checkpoint saved: {ckpt_path}")

    writer.close()
    print(f"\n[Done] Best HR@50={best_metric:.4f}  checkpoint → {best_ckpt_path}")
    return best_ckpt_path

def main(args):
    args.USER_SPARSE = ["121", "122", "124", "125", "126", "127", "128", "129"]
    args.USER_DENSE  = []
    args.ITEM_SPARSE = ["206", "207", "210", "216"]
    args.ITEM_DENSE  = []

    args.vocabulary_size = {
        '101': 238635, '121': 98,     '122': 14,     '124': 3,
        '125': 8,      '126': 4,      '127': 4,      '128': 3,
        '129': 5,      '205': 467298, '206': 6929,   '207': 263942,
        '216': 106399, '508': 5888,   '509': 104830, '702': 51878,
        '853': 37148,  '301': 4,}

    print("loading training set!")

    if args.mode == "pair":
        print("loading training set!")
        train_ds = AliCCPDataset(
            data_path           = args.train_path,
            user_sparse_columns = USER_SPARSE,
            user_dense_columns  = USER_DENSE,
            item_sparse_columns = ITEM_SPARSE,
            item_dense_columns  = ITEM_DENSE,
        )

        print("loading validation set!")
        val_ds = AliCCPDataset(
            data_path           = args.eval_path,
            user_sparse_columns = USER_SPARSE,
            user_dense_columns  = USER_DENSE,
            item_sparse_columns = ITEM_SPARSE,
            item_dense_columns  = ITEM_DENSE,
        )
    else:
        print("loading training set!")
        train_ds = AliCCPPairDataset(
            data_path           = args.train_path,
            user_sparse_columns = USER_SPARSE,
            user_dense_columns  = USER_DENSE,
            item_sparse_columns = ITEM_SPARSE,
            item_dense_columns  = ITEM_DENSE,
        )

        print("loading validation set!")
        val_ds = AliCCPPairDataset(
            data_path           = args.eval_path,
            user_sparse_columns = USER_SPARSE,
            user_dense_columns  = USER_DENSE,
            item_sparse_columns = ITEM_SPARSE,
            item_dense_columns  = ITEM_DENSE,
        )
    
    model = TwoTowerModel(args)

    print("begain training")
    train_loop(
        model         = model,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        mode          = "single",
        epochs        = 20,
        batch_size    = 2048,
        lr            = 1e-3,
        early_stop    = 5,
        save_dir      = "checkpoints",
        log_dir       = "runs",
    ) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Tower Recall Model Training")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path",  type=str, required=True)

    parser.add_argument("--embedding_dim",  type=int,   default=32)
    parser.add_argument("--hidden_dims",    type=int,   nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout",        type=float, default=0.2)

    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=2048)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-5)
    parser.add_argument("--early_stop",     type=int,   default=5)
    parser.add_argument("--mode",           type=str,   default="single", choices=["single", "pair", "infonce"])

    parser.add_argument("--save_dir",       type=str,   default="checkpoints")
    parser.add_argument("--log_dir",        type=str,   default="runs")

    parser.add_argument("--gpu",            type=str,   default="0")
    parser.add_argument("--num_workers",    type=int,   default=4)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)