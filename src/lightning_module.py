import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model import TwoTowerModel, RecallLoss


class TwoTowerLightningModule(pl.LightningModule):
    """PyTorch Lightning 模型模块，保持原有训练逻辑"""

    def __init__(
        self,
        user_num: int,
        item_num: int,
        embed_dim: int,
        hidden_dims: list,
        tower_out_dim: int,
        dropout: float,
        user_feature_dims: list,
        item_feature_dims: list,
        mode: str = "single",  # "single" | "pair" | "list"
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.05,
        max_epochs: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 创建模型参数对象
        class Args:
            pass

        args = Args()
        args.user_num = user_num
        args.item_num = item_num
        args.embed_dim = embed_dim
        args.hidden_dims = hidden_dims
        args.tower_out_dim = tower_out_dim
        args.dropout = dropout
        args.user_feature_dims = user_feature_dims
        args.item_feature_dims = item_feature_dims

        # 初始化模型
        self.model = TwoTowerModel(args)

        # 保存训练模式
        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.max_epochs = max_epochs

        # 初始化损失函数
        if mode == "single":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "pair":
            self.criterion = None  # BPR loss 在 forward 中定义
        else:  # list
            self.criterion = RecallLoss(mode="infonce", temperature=temperature)

    def forward(self, user_features, item_features):
        return self.model(user_features, item_features)

    def training_step(self, batch, batch_idx):
        user_sparse, item_sparse, label = batch
        label = label.float()

        # 前向传播
        if self.mode == "single":
            # Pointwise BCE
            scores = self.model(user_sparse, item_sparse)
            loss = self.criterion(scores, label)

            # 记录指标
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        elif self.mode == "pair":
            # Pairwise BPR (需要成对的正负样本)
            pos_mask = (label == 1)
            neg_mask = (label == 0)

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = self.model(user_sparse[pos_mask], item_sparse[pos_mask])
                neg_scores = self.model(user_sparse[neg_mask], item_sparse[neg_mask])

                # BPR loss
                min_len = min(len(pos_scores), len(neg_scores))
                pos_scores = pos_scores[:min_len]
                neg_scores = neg_scores[:min_len].unsqueeze(1)

                loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()
                self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        else:  # list
            # Listwise InfoNCE (需要重组为正负样本对)
            pos_mask = (label == 1)
            neg_mask = (label == 0)

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = self.model(user_sparse[pos_mask], item_sparse[pos_mask])
                neg_scores = self.model(user_sparse[neg_mask], item_sparse[neg_mask])

                # 重组为 (B, num_neg) 格式
                num_pos = len(pos_scores)
                num_neg = len(neg_scores)
                neg_per_pos = num_neg // num_pos if num_pos > 0 else 0

                if neg_per_pos > 0:
                    neg_scores_reshaped = neg_scores[:num_pos * neg_per_pos].view(num_pos, neg_per_pos)
                    loss = self.criterion(pos_scores[:num_pos], neg_scores_reshaped)
                    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user_sparse, item_sparse, label = batch
        label = label.float()

        # 前向传播
        scores = self.model(user_sparse, item_sparse)

        # 计算验证损失
        if self.mode == "single":
            val_loss = self.criterion(scores, label)
        else:
            # 对于 pair 和 list 模式，使用 BCE 作为验证损失
            val_loss = nn.functional.binary_cross_entropy_with_logits(scores, label)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": val_loss, "scores": scores, "labels": label}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.lr * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    def on_train_epoch_end(self):
        """每个训练epoch结束时重新采样负样本"""
        # 获取数据模块并重新采样
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            self.trainer.datamodule.train_dataset.resample_negatives()
            print(f"\n[Epoch {self.current_epoch + 1}] Resampled negative samples")
