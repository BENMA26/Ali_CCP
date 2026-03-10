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

class AliCCPDataset(Dataset):
    def __init__(self,
        data_path,
        user_sparse_columns,
        user_dense_columns,
        item_sparse_columns,
        item_dense_columns,
        ):
        self.user_sparse_columns = user_sparse_columns
        self.user_dense_columns = user_dense_columns
        self.item_sparse_columns = item_sparse_columns
        self.item_dense_columns = item_dense_columns
        self.df = pd.read_csv(data_path)
        self.user_sparse_features = self.df[user_sparse_columns]
        self.user_dense_features = self.df[user_dense_columns]
        self.item_sparse_features = self.df[item_sparse_columns]
        self.item_dense_features = self.df[item_dense_columns]
        self.label = self.df["click"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user_sparse = self.user_sparse_features.iloc[index].values if self.user_sparse_columns else None
        user_dense = self.user_dense_features.iloc[index].values if self.user_dense_columns else None
        item_sparse = self.item_sparse_features.iloc[index].values if self.item_sparse_columns else None
        item_dense = self.item_dense_features.iloc[index].values if self.item_dense_columns else None
        label = self.label.iloc[index]
        #return user_sparse, user_dense, item_sparse, item_dense, label
        return user_sparse, item_sparse, label


