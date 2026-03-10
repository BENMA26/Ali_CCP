import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TwoTowerModel(torch.nn.Module):
    def __init__(self, args):
        super(TwoTowerModel, self).__init__()
        
        user_num        = args.user_num 
        item_num        = args.item_num 
        embed_dim       = args.embed_dim 
        hidden_dims     = args.hidden_dims 
        tower_out_dim   = args.tower_out_dim 
        dropout         = getattr(args, 'dropout', 0.1)
        
        self.user_embedders = nn.ModuleList([nn.Embedding(feature_dim, embed_dim, padding_idx=0) for feature_dim in args.user_feature_dims])
        self.item_embedders = nn.ModuleList([nn.Embedding(feature_dim, embed_dim, padding_idx=0) for feature_dim in args.item_feature_dims])
        
        user_input_dim = embed_dim * len(args.user_feature_dims)
        item_input_dim = embed_dim * len(args.item_feature_dims)
        
        self.user_tower = Tower(user_input_dim, hidden_dims, tower_out_dim, dropout)
        self.item_tower = Tower(item_input_dim, hidden_dims, tower_out_dim, dropout)

    def forward(self, user_features, item_features):
        user_embed = torch.cat([self.user_embedders[i](user_features[:, i]) for i in range(len(self.user_embedders))], dim=-1)  # [B, embed_dim * num_user_cols]
        item_embed = torch.cat([self.item_embedders[i](item_features[:, i]) for i in range(len(self.item_embedders))], dim=-1)  # [B, embed_dim * num_item_cols]
        
        user_out = self.user_tower(user_embed)      # [B, tower_out_dim]
        item_out = self.item_tower(item_embed)      # [B, tower_out_dim]
        
        user_out = F.normalize(user_out, dim=-1)
        item_out = F.normalize(item_out, dim=-1)
        scores = (user_out * item_out).sum(dim=-1)  # [B]
        
        return scores

    def get_user_embedding(self, user_features):
        user_embed = torch.cat([self.user_embedders[i](user_features[:, i]) for i in range(len(self.user_embedders))], dim=-1)
        return F.normalize(self.user_tower(user_embed), dim=-1)
    
    def get_item_embedding(self, item_features):
        item_embed = torch.cat([self.item_embedders[i](item_features[:, i]) for i in range(len(self.item_embedders))], dim=-1)
        return F.normalize(self.item_tower(item_embed), dim=-1)