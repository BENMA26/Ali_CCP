import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class AliCCPDatasetWithSampling(Dataset):
    """支持负样本动态采样的数据集"""

    def __init__(
        self,
        data_path,
        user_sparse_columns,
        user_dense_columns,
        item_sparse_columns,
        item_dense_columns,
        neg_sample_ratio=1.0,  # 负样本采样比率，1.0表示使用全部负样本
    ):
        self.user_sparse_columns = user_sparse_columns
        self.user_dense_columns = user_dense_columns
        self.item_sparse_columns = item_sparse_columns
        self.item_dense_columns = item_dense_columns
        self.neg_sample_ratio = neg_sample_ratio

        # 读取数据
        self.df = pd.read_csv(data_path)

        # 分离正负样本
        self.pos_df = self.df[self.df["click"] == 1].reset_index(drop=True)
        self.neg_df = self.df[self.df["click"] == 0].reset_index(drop=True)

        print(f"[Dataset] Loaded {len(self.pos_df)} positive samples, {len(self.neg_df)} negative samples")
        print(f"[Dataset] Negative sampling ratio: {neg_sample_ratio}")

        # 计算每个epoch使用的负样本数量
        self.num_neg_samples = int(len(self.neg_df) * neg_sample_ratio)

        # 初始化负样本索引
        self.resample_negatives()

    def resample_negatives(self):
        """重新采样负样本索引"""
        if self.neg_sample_ratio >= 1.0:
            # 使用全部负样本
            self.sampled_neg_indices = np.arange(len(self.neg_df))
        else:
            # 随机采样
            self.sampled_neg_indices = np.random.choice(
                len(self.neg_df),
                size=self.num_neg_samples,
                replace=False
            )
        np.random.shuffle(self.sampled_neg_indices)

    def __len__(self):
        return len(self.pos_df) + len(self.sampled_neg_indices)

    def __getitem__(self, index):
        # 前半部分是正样本，后半部分是负样本
        if index < len(self.pos_df):
            row = self.pos_df.iloc[index]
            label = 1
        else:
            neg_idx = self.sampled_neg_indices[index - len(self.pos_df)]
            row = self.neg_df.iloc[neg_idx]
            label = 0

        user_sparse = row[self.user_sparse_columns].values if self.user_sparse_columns else None
        item_sparse = row[self.item_sparse_columns].values if self.item_sparse_columns else None

        return user_sparse, item_sparse, label


class AliCCPDataModule(pl.LightningDataModule):
    """PyTorch Lightning 数据模块"""

    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        user_sparse_columns: list,
        user_dense_columns: list,
        item_sparse_columns: list,
        item_dense_columns: list,
        batch_size: int = 2048,
        num_workers: int = 4,
        neg_sample_ratio: float = 1.0,  # 负样本采样比率
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.user_sparse_columns = user_sparse_columns
        self.user_dense_columns = user_dense_columns
        self.item_sparse_columns = item_sparse_columns
        self.item_dense_columns = item_dense_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_ratio = neg_sample_ratio

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            print("Loading training set!")
            self.train_dataset = AliCCPDatasetWithSampling(
                data_path=self.train_data_path,
                user_sparse_columns=self.user_sparse_columns,
                user_dense_columns=self.user_dense_columns,
                item_sparse_columns=self.item_sparse_columns,
                item_dense_columns=self.item_dense_columns,
                neg_sample_ratio=self.neg_sample_ratio,
            )

            print("Loading validation set!")
            self.val_dataset = AliCCPDatasetWithSampling(
                data_path=self.val_data_path,
                user_sparse_columns=self.user_sparse_columns,
                user_dense_columns=self.user_dense_columns,
                item_sparse_columns=self.item_sparse_columns,
                item_dense_columns=self.item_dense_columns,
                neg_sample_ratio=1.0,  # 验证集使用全部负样本
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """每个epoch结束后重新采样负样本"""
        return batch
