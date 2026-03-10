# ------ User Features 用户特征 ------
# 101     : User ID
#           用户ID
# sequence data type
# 109_14  : User historical behaviors of category ID and count*
#           用户历史行为：类目ID及其交互次数（14天）
# 110_14  : User historical behaviors of shop ID and count*
#           用户历史行为：店铺ID及其交互次数（14天）
# 127_14  : User historical behaviors of brand ID and count*
#           用户历史行为：品牌ID及其交互次数（14天）
# 150_14  : User historical behaviors of intention node ID and count*
#           用户历史行为：意图节点ID及其交互次数（14天）
# profile data type
# 121     : Categorical ID of User Profile
#           用户画像类别ID
# 122     : Categorical group ID of User Profile
#           用户画像类别组ID
# 124     : Users Gender ID
#           用户性别ID
# 125     : Users Age ID
#           用户年龄ID
# 126     : Users Consumption Level Type I
#           用户消费等级（类型I）
# 127     : Users Consumption Level Type II
#           用户消费等级（类型II）
# 128     : Users Occupation: whether or not to work
#           用户职业：是否在职
# 129     : Users Geography Informations
#           用户地理位置信息
# ------ Item Features 商品特征 ------
# item profile
# 205     : Item ID
#           商品ID
# 206     : Category ID to which the item belongs to
#           商品所属类目ID
# 207     : Shop ID to which item belongs to
#           商品所属店铺ID
# 210     : Intention node ID which the item belongs to
#           商品所属意图节点ID
# 216     : Brand ID of the item
#           商品品牌ID
# ------ Combination Features 组合特征（用户-商品交叉）------
# 508     : The combination of features with 109_14 and 206
#           用户类目行为(109_14) × 商品类目(206) 的交叉统计
# 509     : The combination of features with 110_14 and 207
#           用户店铺行为(110_14) × 商品店铺(207) 的交叉统计
# 702     : The combination of features with 127_14 and 216
#           用户品牌行为(127_14) × 商品品牌(216) 的交叉统计
# 853     : The combination of features with 150_14 and 210
#           用户意图行为(150_14) × 商品意图节点(210) 的交叉统计
# ------ Context Features 场景特征 ------
# 301     : A categorical expression of position
#           广告位置的类别表达（广告位类型）

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ─────────────────────────────────────────────
# 策略1: 随机负采样 Dataset（离线生成负样本）
# ─────────────────────────────────────────────
class AliCCPDatasetWithNegSampling(Dataset):
    """
    在加载时为每条正样本随机采样 num_neg 条负样本。
    负样本 = 对该 user 未曾点击过的 item。
    """
    def __init__(
        self,
        data_path,
        user_sparse_columns,
        user_dense_columns,
        item_sparse_columns,
        item_dense_columns,
        item_pool_path=None,   # 全量 item 特征表 CSV，若 None 则从数据集自身构建
        num_neg: int = 4,
        neg_strategy: str = "random",  # "random" | "popularity"
        seed: int = 42,
    ):
        self.data = pd.read_csv(data_path)
        self.user_sparse_columns = user_sparse_columns
        self.user_dense_columns  = user_dense_columns
        self.item_sparse_columns = item_sparse_columns
        self.item_dense_columns  = item_dense_columns
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

        # ── 构建 item 特征池 ──────────────────────────────────
        if item_pool_path:
            item_df = pd.read_csv(item_pool_path)
        else:
            # 从数据集中去重得到 item 特征池
            item_df = self.data[item_sparse_columns + item_dense_columns].drop_duplicates()

        self.item_pool = item_df.reset_index(drop=True)
        self.item_ids  = np.arange(len(self.item_pool))  # 用行索引代表 item id

        # ── 构建 user -> 正样本 item 行索引集合（用于排除）────
        # 用 item_sparse_columns 拼接字符串作为 item 唯一 key
        self.data["_item_key"] = (
            self.data[item_sparse_columns]
            .astype(str)
            .agg("_".join, axis=1)
        )
        self.item_pool["_item_key"] = (
            self.item_pool[item_sparse_columns]
            .astype(str)
            .agg("_".join, axis=1)
        )
        key2idx = dict(zip(self.item_pool["_item_key"], self.item_pool.index))
        self.data["_item_idx"] = self.data["_item_key"].map(key2idx)

        # user_key -> set of positive item indices
        user_col = user_sparse_columns[0]  # 用第一个稀疏列作 user id
        self.user_pos_items: dict = (
            self.data.groupby(user_col)["_item_idx"]
            .apply(set)
            .to_dict()
        )

        # ── 流行度权重（可选）────────────────────────────────
        if neg_strategy == "popularity":
            cnt = Counter(self.data["_item_idx"].tolist())
            freq = np.array([cnt.get(i, 0) for i in self.item_ids], dtype=np.float64)
            # 平滑：frequency^0.75（Word2Vec 风格）
            freq = np.power(freq + 1, 0.75)
            self.neg_weights = freq / freq.sum()
        else:
            self.neg_weights = None  # 均匀采样

        # 只保留正样本行（click=1）
        self.pos_data = self.data[self.data["click"] == 1].reset_index(drop=True)

    def __len__(self):
        return len(self.pos_data)

    def _sample_negatives(self, user_key, n):
        """为 user_key 采 n 个负样本 item 的行索引（排除正样本）"""
        pos_set = self.user_pos_items.get(user_key, set())
        candidates = np.setdiff1d(self.item_ids, list(pos_set))

        if len(candidates) == 0:
            candidates = self.item_ids  # 极端情况退化为全集

        if self.neg_weights is not None:
            weights = self.neg_weights[candidates]
            weights = weights / weights.sum()
        else:
            weights = None

        chosen = self.rng.choice(candidates, size=n, replace=False, p=weights)
        return chosen

    def _row_to_item_tensors(self, item_row):
        sparse = torch.tensor(
            [item_row[col] for col in self.item_sparse_columns], dtype=torch.long
        )
        dense = (
            torch.tensor([item_row[col] for col in self.item_dense_columns], dtype=torch.float32)
            if self.item_dense_columns
            else torch.zeros(0)
        )
        return sparse, dense

    def __getitem__(self, idx):
        row = self.pos_data.iloc[idx]
        user_key = row[self.user_sparse_columns[0]]

        # ── 用户特征 ────────────────────────────────────────
        user_sparse = torch.tensor(
            [row[col] for col in self.user_sparse_columns], dtype=torch.long
        )
        user_dense = (
            torch.tensor([row[col] for col in self.user_dense_columns], dtype=torch.float32)
            if self.user_dense_columns
            else torch.zeros(0)
        )

        # ── 正样本 item ─────────────────────────────────────
        pos_item_sparse, pos_item_dense = self._row_to_item_tensors(row)

        # ── 负样本 items ────────────────────────────────────
        neg_idxs = self._sample_negatives(user_key, self.num_neg)
        neg_sparse_list, neg_dense_list = [], []
        for nidx in neg_idxs:
            ns, nd = self._row_to_item_tensors(self.item_pool.iloc[nidx])
            neg_sparse_list.append(ns)
            neg_dense_list.append(nd)

        neg_item_sparse = torch.stack(neg_sparse_list)  # (num_neg, num_item_sparse)
        neg_item_dense  = torch.stack(neg_dense_list)   # (num_neg, num_item_dense)

        return {
            # 用户侧
            "user_sparse":      user_sparse,        # (num_user_sparse,)
            "user_dense":       user_dense,          # (num_user_dense,)
            # 正样本
            "pos_item_sparse":  pos_item_sparse,     # (num_item_sparse,)
            "pos_item_dense":   pos_item_dense,      # (num_item_dense,)
            # 负样本
            "neg_item_sparse":  neg_item_sparse,     # (num_neg, num_item_sparse)
            "neg_item_dense":   neg_item_dense,      # (num_neg, num_item_dense)
        }

# ─────────────────────────────────────────────
# 策略2: In-Batch 负采样 Collator（动态，零额外开销）
# ─────────────────────────────────────────────
class InBatchNegCollator:
    """
    将同一 batch 内其他样本的 item 作为负样本，
    不需要修改 Dataset，只替换 DataLoader 的 collate_fn。
    适配原始 AliCCPDataset（只含正负标签，不额外采样）。
    """

    def __call__(self, batch):
        # batch: List[dict]，每个 dict 来自 AliCCPDataset.__getitem__
        keys = batch[0].keys()
        collated = {k: torch.stack([b[k] for b in batch]) for k in keys}

        B = len(batch)
        # item_sparse: (B, num_item_sparse)，item_dense: (B, num_item_dense)
        item_sparse = collated["item_sparse"]  # 正样本 item
        item_dense  = collated["item_dense"]

        # 构造 In-Batch 负样本矩阵（每个 query 把其他 B-1 条当负样本）
        # neg_item_sparse: (B, B-1, num_item_sparse)
        neg_sparse_list, neg_dense_list = [], []
        for i in range(B):
            neg_idx = [j for j in range(B) if j != i]
            neg_sparse_list.append(item_sparse[neg_idx])  # (B-1, F)
            neg_dense_list.append(item_dense[neg_idx])

        collated["neg_item_sparse"] = torch.stack(neg_sparse_list)  # (B, B-1, F_sparse)
        collated["neg_item_dense"]  = torch.stack(neg_dense_list)   # (B, B-1, F_dense)
        return collated

# ─────────────────────────────────────────────
# 召回双塔损失（BPR + InfoNCE 可选）
# ─────────────────────────────────────────────
class RecallLoss(torch.nn.Module):
    """
    支持两种损失：
      - 'bpr'    : sum_neg log σ(s_pos - s_neg)
      - 'infonce': -log [ exp(s_pos/τ) / (exp(s_pos/τ) + Σ exp(s_neg/τ)) ]
    """

    def __init__(self, mode: str = "infonce", temperature: float = 0.05):
        super().__init__()
        assert mode in ("bpr", "infonce")
        self.mode = mode
        self.tau  = temperature

    def forward(self, pos_score: torch.Tensor, neg_scores: torch.Tensor):
        """
        pos_score : (B,)
        neg_scores: (B, num_neg)
        """
        if self.mode == "bpr":
            diff = pos_score.unsqueeze(1) - neg_scores      # (B, num_neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        else:  # infonce
            pos  = (pos_score / self.tau).unsqueeze(1)      # (B, 1)
            negs = neg_scores / self.tau                     # (B, num_neg)
            logits = torch.cat([pos, negs], dim=1)           # (B, 1+num_neg)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss




