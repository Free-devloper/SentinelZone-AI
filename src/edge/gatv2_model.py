import torch
import torch.nn as nn
from typing import Optional

try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

    class GATv2Conv(nn.Module):
        """
        Native PyTorch fallback for GATv2Conv with edge attributes when torch_geometric
        is not available in the local runtime.
        """
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 4,
            concat: bool = False,
            edge_dim: int = 64,
            dropout: float = 0.0
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat
            self.edge_dim = edge_dim
            self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att = nn.Parameter(torch.empty(1, heads, out_channels))
            nn.init.xavier_uniform_(self.att)
            if not concat:
                self.out_proj = nn.Linear(heads * out_channels, out_channels)
            else:
                self.out_proj = nn.Identity()

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
            N = x.size(0)
            E = edge_index.size(1)

            src, dst = edge_index[0], edge_index[1]
            x_l = self.lin_l(x).view(N, self.heads, self.out_channels)
            x_r = self.lin_r(x).view(N, self.heads, self.out_channels)
            e_feat = self.lin_edge(edge_attr).view(E, self.heads, self.out_channels)

            # GATv2 scoring
            alpha = torch.nn.functional.leaky_relu(x_l[src] + x_r[dst] + e_feat, negative_slope=0.2)
            alpha = (alpha * self.att).sum(dim=-1)

            # Vectorized softmax normalization per destination node
            exp_alpha = torch.exp(alpha - torch.clamp(alpha.max(dim=0, keepdim=True)[0], min=-10.0, max=10.0))
            sum_exp = torch.zeros((N, self.heads), device=x.device, dtype=x.dtype)
            sum_exp = sum_exp.index_add(0, dst, exp_alpha) + 1e-6
            norm_alpha = exp_alpha / (sum_exp[dst] + 1e-6)

            # Weighted message passing and aggregation
            msg = (x_l[src] + e_feat) * norm_alpha.unsqueeze(-1)
            out = torch.zeros((N, self.heads, self.out_channels), device=x.device, dtype=x.dtype)
            out = out.index_add(0, dst, msg)

            out_flat = out.view(N, self.heads * self.out_channels)
            return self.out_proj(out_flat)


class WorkZoneSTGNN(nn.Module):
    """
    Spatio-Temporal Interaction Graph Neural Network.
    Encodes temporal agent histories via GRU, model multi-agent interactions via GATv2,
    and outputs multimodal future trajectories parameterized as Gaussian Mixture Models.
    """
    def __init__(
        self,
        node_features_dim: int = 8,
        class_dim: int = 16,
        edge_features_dim: int = 5,
        hidden_dim: int = 128,
        num_heads: int = 4,
        pred_steps: int = 50,
        num_modes: int = 3
    ):
        super().__init__()
        self.pred_steps = pred_steps
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        self.class_embedding = nn.Embedding(16, class_dim)

        self.temporal_encoder = nn.GRU(
            input_size=node_features_dim + class_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.gatv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            edge_dim=64,
            dropout=0.0
        )

        self.mode_head = nn.Linear(hidden_dim, num_modes)
        self.trajectory_head = nn.Linear(hidden_dim, num_modes * pred_steps * 5)

    def forward(
        self, 
        node_history: torch.Tensor, 
        class_ids: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor
    ) -> dict:
        """
        node_history: [N, T_obs=30, 8]
        class_ids:    [N]
        edge_index:   [2, E]
        edge_attr:    [E, 5]
        """
        N, T_obs, _ = node_history.shape
        cls_emb = self.class_embedding(class_ids).unsqueeze(1).repeat(1, T_obs, 1)
        node_in = torch.cat([node_history, cls_emb], dim=-1)

        _, h_n = self.temporal_encoder(node_in)
        node_state = h_n[-1]

        edge_emb = self.edge_encoder(edge_attr)
        interaction_state = self.gatv2(node_state, edge_index, edge_emb)
        fused = node_state + interaction_state

        mode_probs = torch.softmax(self.mode_head(fused), dim=-1)
        raw_trajs = self.trajectory_head(fused).view(N, self.num_modes, self.pred_steps, 5)

        mu_x = raw_trajs[..., 0]
        mu_y = raw_trajs[..., 1]
        sigma_x = torch.exp(raw_trajs[..., 2]) + 1e-4
        sigma_y = torch.exp(raw_trajs[..., 3]) + 1e-4
        rho = torch.tanh(raw_trajs[..., 4])

        return {
            "mode_probs": mode_probs,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho
        }
