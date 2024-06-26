import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(PolicyNetwork, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, hidden_dim * 4)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        x = self.transformer_layer(x)
        x = x.squeeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.policy_head(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(ValueNetwork, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, hidden_dim * 4)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        x = self.transformer_layer(x)
        x = x.squeeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value
