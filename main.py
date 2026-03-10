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
from src.dataset import AliCCPDataset
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"[Checkpoint] saved → {path}")

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt["epoch"]

@torch.no_grad()
def compute_metrics(user_emb: torch.Tensor,
                    pos_item_emb: torch.Tensor,
                    neg_item_emb: torch.Tensor,
                    ks=(20, 50, 100)):
    """
    user_emb     : (B, D)
    pos_item_emb : (B, D)          — 每个 user 对应 1 条正样本
    neg_item_emb : (B, num_neg, D) — 每个 user 对应 num_neg 条负样本
    返回 HitRate@k 和 NDCG@k
    """
    B, D = user_emb.shape
    # 拼正+负 → (B, 1+num_neg, D)
    all_items = torch.cat([pos_item_emb.unsqueeze(1), neg_item_emb], dim=1)

    # 计算相似度分数 (B, 1+num_neg)
    scores = torch.bmm(all_items, user_emb.unsqueeze(-1)).squeeze(-1)

    # 排序（降序），正样本在 index=0 位置
    ranks = (scores > scores[:, 0:1]).sum(dim=1)  # 正样本的排名（0-indexed）

    metrics = {}
    for k in ks:
        hit  = (ranks < k).float().mean().item()
        # NDCG@k: 命中时 1/log2(rank+2)，未命中为 0
        ndcg = (ranks < k).float() * (1.0 / torch.log2(ranks.float() + 2))
        metrics[f"HR@{k}"]   = hit
        metrics[f"NDCG@{k}"] = ndcg.mean().item()

    return metrics

def forward_batch(model, batch, criterion, device, mode):
    """
    返回 (loss, pos_score, neg_scores, user_emb, pos_emb, neg_emb)

    mode="single" : Pointwise，直接对正/负样本做二分类
    mode="pair"   : Pairwise BPR，每条正样本对应 1 条负样本
    mode="list"   : Listwise InfoNCE，每条正样本对应 num_neg 条负样本
    """
    def to(x): return x.to(device)
    user_sparse, item_sparse, label = batch
    user_sparse = to(user_sparse)
    item_sparse = to(item_sparse)
    label = to(label)

    output = model(user_sparse,item_sparse)

    if mode == "single":
        score = output
        label = label.float()

        loss = criterion(score, label)               # BCEWithLogitsLoss

        # 区分正负样本，供 compute_metrics 使用
        pos_mask = (label == 1)
        neg_mask = (label == 0)

        pos_score  = score[pos_mask] 
        neg_scores = score[neg_mask]

        return loss, pos_score, neg_scores

def train_loop(
    model,
    train_dataset,
    val_dataset,
    mode: str = "list",           # "single" | "pair" | "list"
    # ── 超参 ──────────────────────────────────────────────────
    epochs: int         = 20,
    batch_size: int     = 2048,
    lr: float           = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float  = 0.05,   # InfoNCE 温度（list 模式）
    # ── 工程配置 ─────────────────────────────────────────────
    num_workers: int    = 4,
    device_str: str     = "auto",
    save_dir: str       = "checkpoints",
    log_dir: str        = "runs",
    save_every: int     = 1,       # 每 N epoch 保存一次
    eval_every: int     = 1,       # 每 N epoch 评估一次
    early_stop: int     = 5,       # patience（-1 关闭）
    resume_path: str    = None,    # 断点续训路径
    seed: int           = 42,
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # ── 设备 ─────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Device] {device}  |  Mode: {mode}")

    # ── DataLoader ───────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,           # 保证 batch 大小一致（In-Batch 负采样需要）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.to(device)

    # ── 损失函数 ─────────────────────────────────────────────
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

    # ── 优化器 & 调度器 ───────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── TensorBoard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{mode}_{int(time.time())}"))

    # ── 断点续训 ─────────────────────────────────────────────
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(model, optimizer, resume_path, device)

    # ── Early Stopping 状态 ──────────────────────────────────
    best_metric   = -float("inf")   # 以 HR@50 为主指标
    patience_cnt  = 0
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")

    # ═════════════════════════════════════════════════════════
    # Epoch 循环
    # ═════════════════════════════════════════════════════════
    for epoch in range(start_epoch, epochs):
        # ── Train ─────────────────────────────────────────────
        model.train()
        total_loss    = 0.0
        total_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in pbar:
            optimizer.zero_grad()

            loss, pos_score, neg_scores = forward_batch(model, batch, criterion, device, mode)

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

        # ── 定期保存 ─────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, path)

    writer.close()
    print(f"\n[Done] Best HR@50={best_metric:.4f}  checkpoint → {best_ckpt_path}")
    return best_ckpt_path

if __name__ == "__main__":
    USER_SPARSE = ["121", "122", "124", "125", "126", "127", "128", "129"]
    USER_DENSE  = []
    ITEM_SPARSE = ["205", "206", "207", "210", "216"]
    ITEM_DENSE  = []

    print("loading training set!")
    train_ds = AliCCPDataset(
        data_path           = "/work/home/maben/project/rec_sys/projects/Ali_CCP/datasets/datasetsali_ccp_train.csv",
        user_sparse_columns = USER_SPARSE,
        user_dense_columns  = USER_DENSE,
        item_sparse_columns = ITEM_SPARSE,
        item_dense_columns  = ITEM_DENSE,
    )

    print("loading validation set!")
    val_ds = AliCCPDataset(
        data_path           = "/work/home/maben/project/rec_sys/projects/Ali_CCP/datasets/datasetsali_ccp_val.csv",
        user_sparse_columns = USER_SPARSE,
        user_dense_columns  = USER_DENSE,
        item_sparse_columns = ITEM_SPARSE,
        item_dense_columns  = ITEM_DENSE,
    )

    args = argparse.Namespace(
        user_num        = 1000,
        item_num        = 5000,
        embed_dim       = 16,
        hidden_dims     = [128, 64],
        tower_out_dim   = 32,
        dropout         = 0.1,
        user_feature_dims = [98, 14, 3, 8, 4, 4, 3, 5],   # 每个user列的vocab_size
        item_feature_dims = [538376, 7092, 285825, 82412, 113262],           # 每个item列的vocab_size
    )

    model = TwoTowerModel(args)

    print("begain training")
    train_loop(
        model         = model,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        mode          = "single",    # 推荐 InfoNCE
        epochs        = 20,
        batch_size    = 2048,
        lr            = 1e-3,
        early_stop    = 5,
        save_dir      = "checkpoints",
        log_dir       = "runs",
    )