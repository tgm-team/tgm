from typing import Optional, Tuple

import torch


def _sparse_sliding(
    adj: torch.Tensor,
    rows: torch.Tensor | None = None,
    cols: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Slicing 2D Sparse Tensor.

    Args:
        adj (torch.Tensor) : Sparse tensor for slicing
        rows (torch.Tensor) : Index of selected rows
        cols (torch.Tensor) : Index of selected cols
    """
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    new_row = adj.size(0)
    new_col = adj.size(1)

    if rows is not None and rows.numel() > 0:
        new_row = rows.numel()
        mask = torch.isin(indices[0], rows)
        indices = indices[:, mask]
        values = values[mask]

        mapping = torch.full(
            (int(rows.max() + 1),), -1, device=adj.device, dtype=torch.long
        )
        mapping[rows] = torch.arange(len(rows), device=adj.device)
        indices[0] = mapping[indices[0]]

    if cols is not None and cols.numel() > 0:
        new_col = cols.numel()
        mask = torch.isin(indices[1], cols)
        indices = indices[:, mask]
        values = values[mask]

        mapping = torch.full(
            (int(cols.max() + 1),), -1, device=adj.device, dtype=torch.long
        )
        mapping[cols] = torch.arange(len(cols), device=adj.device)
        indices[1] = mapping[indices[1]]

    return torch.sparse_coo_tensor(
        indices, values, size=(new_row, new_col), device=adj.device
    ).coalesce()


def _fill(sp: torch.Tensor, value: int | float) -> torch.Tensor:
    r"""Fill torch.sparse_coo_tensor with given value."""
    return torch.sparse_coo_tensor(
        sp.indices(),
        torch.full((sp.values().numel(),), value),
        sp.size(),
        device=sp.device,
    ).coalesce()


class NCNPredictor(
    torch.nn.Module
):  # @TODO: Added torch-sparse version when torch-sparse is supported in TGM.
    r"""An implementation of Temporal Neural Common Neighbor (TNCN).

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_dim (int): Size of each hidden embedding.
        k (int): define k-th hop common neighbour (CN) embedding extraction (select from 2/4/8)
        cn_time_decay (bool): indicate whether applying decay on time

    Reference: https://arxiv.org/abs/2406.07926.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        k: int = 2,
        cn_time_decay: bool = False,
    ) -> None:
        super().__init__()
        if k not in [2, 4, 8]:
            raise ValueError('Please choose k from [2,4,8]')

        self.k = k
        self.xslin = torch.nn.Linear(k * in_channels, out_channels)
        self.xsmlp = torch.nn.Sequential(
            torch.nn.Linear(k * in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels),
        )
        self.cn_time_decay = cn_time_decay

    def get_cn_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Obtain the CNs embeddings for each node pair in a batch (Torch version).

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edges list of subgraph,
            tar_ei (torch.Tensor): edges list for prediction,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        tar_i, tar_j = tar_ei[0], tar_ei[1]
        if self.cn_time_decay:
            if time_info is None:
                raise RuntimeError(
                    'Please provide time_information to perform time decay'
                )
            last_update, pos_t = time_info
            last_update = last_update.unsqueeze(0)  # 1*N
            pos_t = pos_t.unsqueeze(1)  # B*1
            time_decay_matrix = (pos_t - last_update) / 10000  # time scale
            time_decay_matrix = torch.exp(-time_decay_matrix)

        id_num = x.size(0)
        adj1 = (
            torch.sparse_coo_tensor(
                torch.cat(
                    (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                    dim=-1,
                ),
                torch.ones(edge_index.shape[1] * 2, device=x.device),
                size=(id_num, id_num),
            )
            .coalesce()
            .to(x.device)
        )

        if self.k == 4:
            indices = torch.arange(id_num, device=x.device)
            adj0 = torch.sparse_coo_tensor(
                torch.stack([indices, indices], dim=0),
                torch.ones(id_num, device=x.device),
                size=(id_num, id_num),
                device=x.device,
            )

            i_0_v, i_1_v, j_0_v, j_1_v = (
                _sparse_sliding(adj0, tar_i),
                _sparse_sliding(adj1, tar_i),
                _sparse_sliding(adj0, tar_j),
                _sparse_sliding(adj1, tar_j),
            )

            i_0_e, i_1_e, j_0_e, j_1_e = (
                _fill(i_0_v, 1.0),
                _fill(i_1_v, 1.0),
                _fill(j_0_v, 1.0),
                _fill(j_1_v, 1.0),
            )

            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v

            if self.cn_time_decay:
                cn_0_1, cn_1_0, cn_1_1 = (
                    cn_0_1 * time_decay_matrix,
                    cn_1_0 * time_decay_matrix,
                    cn_1_1 * time_decay_matrix,
                )
            xcn_0_1, xcn_1_0, xcn_1_1 = (
                torch.sparse.mm(cn_0_1, x),
                torch.sparse.mm(cn_1_0, x),
                torch.sparse.mm(cn_1_1, x),
            )
            cn_emb = torch.cat([xcn_0_1, xcn_1_0, xcn_1_1], dim=-1)

        elif self.k == 2:
            i_1_v, j_1_v = (
                _sparse_sliding(adj1, tar_i),
                _sparse_sliding(adj1, tar_j),
            )
            i_1_e, j_1_e = _fill(i_1_v, 1.0), _fill(j_1_v, 1.0)
            cn_1_1 = i_1_v * j_1_v
            if self.cn_time_decay:
                cn_1_1 = cn_1_1 * time_decay_matrix
            xcn_1_1 = torch.sparse.mm(cn_1_1, x)
            cn_emb = torch.cat([xcn_1_1], dim=-1)

        elif self.k == 8:
            indices = torch.arange(id_num, device=x.device)
            adj0 = torch.sparse_coo_tensor(
                torch.stack([indices, indices], dim=0),
                torch.ones(id_num, device=x.device),
                size=(id_num, id_num),
                device=x.device,
            )

            adj2 = torch.sparse.mm(adj1, adj1)  # self: fake 2 hop
            k3cycle = torch.sparse.mm(adj2, adj1)
            i_0_v, i_1_v, i_2_v, j_0_v, j_1_v, j_2_v = (
                _sparse_sliding(adj0, tar_i),
                _sparse_sliding(adj1, tar_i),
                _sparse_sliding(adj2, tar_i),
                _sparse_sliding(adj0, tar_j),
                _sparse_sliding(adj1, tar_j),
                _sparse_sliding(adj2, tar_j),
            )

            i_0_e, i_1_e, i_2_e, j_0_e, j_1_e, j_2_e = (
                _fill(i_0_v, 1.0),
                _fill(i_1_v, 1.0),
                _fill(i_2_v, 1.0),
                _fill(j_0_v, 1.0),
                _fill(j_1_v, 1.0),
                _fill(j_2_v, 1.0),
            )

            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v
            cn_1_2, cn_2_1, cn_2_2 = (
                (i_1_v * j_2_v),
                (i_2_v * j_1_v),
                (i_2_v * j_2_v),
            )

            u_v_value = _sparse_sliding(adj1, tar_i, tar_j).to_dense().diag().reshape(
                -1, 1
            ) * (-1)
            delta_1_2 = i_1_v * i_1_v * u_v_value
            delta_2_1 = j_1_v * j_1_v * u_v_value
            neg_cn_1_1 = torch.sparse_coo_tensor(
                cn_1_1.indices(),
                cn_1_1.values() * -1,
                cn_1_1.size(),
                device=x.device,
            )
            delta_2_2 = (
                i_1_e
                * _sparse_sliding(k3cycle, tar_i, tar_i)
                .to_dense()
                .diag()
                .reshape(-1, 1)
                + j_1_e
                * _sparse_sliding(k3cycle, tar_j, tar_j)
                .to_dense()
                .diag()
                .reshape(-1, 1)
                + neg_cn_1_1
            ) * u_v_value
            special_2_2 = torch.sparse.mm(cn_1_1, adj1)
            delta_2_2 = delta_2_2 + special_2_2

            cn_1_2, cn_2_1 = cn_1_2 + delta_1_2, cn_2_1 + delta_2_1
            cn_2_2 = cn_2_2 + delta_2_2
            idx = torch.arange(0, len(tar_i), device=x.device).repeat(2)
            u_v_mask = torch.cat([tar_i, tar_j], dim=0)

            cn_1_2, cn_2_1, cn_2_2 = (
                cn_1_2.to_dense(),
                cn_2_1.to_dense(),
                cn_2_2.to_dense(),
            )
            cn_1_2[idx, u_v_mask] = 0
            cn_2_1[idx, u_v_mask] = 0
            cn_2_2[idx, u_v_mask] = 0
            cn_2_2[cn_2_2 < 0] = 0

            if self.cn_time_decay:
                cn_0_1, cn_1_0, cn_1_1 = (
                    cn_0_1.to_dense(),
                    cn_1_0.to_dense(),
                    cn_1_1.to_dense(),
                )
                cn_0_1, cn_1_0, cn_1_1, cn_1_2, cn_2_1, cn_2_2 = (
                    cn_0_1 * time_decay_matrix,
                    cn_1_0 * time_decay_matrix,
                    cn_1_1 * time_decay_matrix,
                    cn_1_2 * time_decay_matrix,
                    cn_2_1 * time_decay_matrix,
                    cn_2_2 * time_decay_matrix,
                )
                cn_0_1, cn_1_0, cn_1_1 = (
                    cn_0_1.to_sparse_coo(),
                    cn_1_0.to_sparse_coo(),
                    cn_1_1.to_sparse_coo(),
                )
            cn_1_2, cn_2_1, cn_2_2 = (
                cn_1_2.to_sparse_coo(),
                cn_2_1.to_sparse_coo(),
                cn_2_2.to_sparse_coo(),
            )
            xcn_0_1, xcn_1_0, xcn_1_1, xcn_1_2, xcn_2_1, xcn_2_2 = (
                torch.sparse.mm(cn_0_1, x),
                torch.sparse.mm(cn_1_0, x),
                torch.sparse.mm(cn_1_1, x),
                torch.sparse.mm(cn_1_2, x),
                torch.sparse.mm(cn_2_1, x),
                torch.sparse.mm(cn_2_2, x),
            )
            special_xcn_2_2 = torch.sparse.mm(special_2_2, x)
            cn_emb = torch.cat(
                [
                    xcn_0_1,
                    xcn_1_0,
                    xcn_1_1,
                    xcn_1_2,
                    xcn_2_1,
                    xcn_2_2,
                    special_xcn_2_2,
                ],
                dim=-1,
            )

        return cn_emb

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edges list of subgraph,
            tar_ei (torch.Tensor): edges list for prediction ,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]

        xij = torch.mul(xi, xj).reshape(-1, x.size(1))
        cn_emb = self.get_cn_emb(x, edge_index, tar_ei, time_info)
        xs = torch.cat([xij, cn_emb], dim=-1)

        xs.relu()
        xs = self.xsmlp(xs)

        return xs.view(-1)
