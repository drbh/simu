import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=4):
        super(PolicyNetwork, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, hidden_dim * 4)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(0)
        mean = self.fc(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_heads=4):
        super(ValueNetwork, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, hidden_dim * 4)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.squeeze(0)
        return self.fc(x)
