import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(Tower, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class Two_Tower_Model(torch.nn.Module):
    def __init__(self, args):
        super(Two_Tower_Model, self).__init__()
        
        user_num        = args.user_num         # 用户数量
        item_num        = args.item_num         # 物品数量
        embed_dim       = args.embed_dim        # embedding维度
        hidden_dims     = args.hidden_dims      # 隐藏层维度列表, 如 [256, 128]
        tower_out_dim   = args.tower_out_dim    # 塔输出维度
        dropout         = getattr(args, 'dropout', 0.1)
        
        # Embedding 层
        self.user_embedder = nn.Embedding(user_num, embed_dim, padding_idx=0)
        self.item_embedder = nn.Embedding(item_num, embed_dim, padding_idx=0)
        
        # 双塔
        self.user_tower = Tower(embed_dim, hidden_dims, tower_out_dim, dropout)
        self.item_tower = Tower(embed_dim, hidden_dims, tower_out_dim, dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedder.weight)
        nn.init.xavier_uniform_(self.item_embedder.weight)

    def forward(self, user_ids, item_ids):
        # Embedding
        user_embed = self.user_embedder(user_ids)   # [B, embed_dim]
        item_embed = self.item_embedder(item_ids)   # [B, embed_dim]
        
        # 各自过塔
        user_out = self.user_tower(user_embed)      # [B, tower_out_dim]
        item_out = self.item_tower(item_embed)      # [B, tower_out_dim]
        
        # 点积 (召回场景，可向量化检索)
        user_out = F.normalize(user_out, dim=-1) # l2 norm
        item_out = F.normalize(item_out, dim=-1) # l2 norm
        scores = (user_out * item_out).sum(dim=-1)  # [B]
        
        return scores

    def get_user_embedding(self, user_ids):
        user_embed = self.user_embedder(user_ids)
        return F.normalize(self.user_tower(user_embed), dim=-1)
    
    def get_item_embedding(self, item_ids):
        item_embed = self.item_embedder(item_ids)
        return F.normalize(self.item_tower(item_embed), dim=-1)