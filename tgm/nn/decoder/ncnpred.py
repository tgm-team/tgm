from typing import Optional, Tuple

import torch
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_add

def _sparse_row_select(adj:torch.Tensor, rows:torch.Tensor) -> torch.Tensor:
    r"""
    Row sliding on torch.sparse_coo_tensor
    """
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    mask = torch.isin(indices[0],rows)
    indices_slice = indices[:,mask]
    values_slice = values[mask]

    mapping = torch.full((rows.max() + 1,),-1,device=adj.device)
    mapping[rows] = torch.arange(len(rows),device=adj.device)
    indices_slice[0]= mapping[indices_slice[0]]
    return torch.sparse_coo_tensor(
        indices_slice,
        values_slice,
        size=(rows.numel(),adj.size(1)),
        device=adj.device
    ).coalesce()
    

def _fill(sp: torch.Tensor, value: int | float) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        sp.indices(),
        torch.full((sp.values().numel(),),value),
        sp.size(),
        device=sp.device,
    ).coalesce()


class NCNPredictor_Sparse(torch.nn.Module):
    r"""An implementation of Temporal Neural Common Neighbor (TNCN).

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_dim (int): Size of each hidden embedding.
        k (int): ...

    Reference: https://arxiv.org/abs/2406.07926.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        k: int = 2,
    ):
        super().__init__()
        assert k in [2, 4, 8], 'Please choose k values from [2,4,8]'

        self.k = k
        self.xslin = torch.nn.Linear(
            k * in_channels, out_channels
        )  # TODO: add more layers
        self.xsmlp = torch.nn.Sequential(
            torch.nn.Linear(k * in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels),
        )

    def get_cn_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        cn_time_decay: bool = False,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Obtain the CNs embeddings for each node pair in a batch.

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edge list,
            tar_ei (torch.Tensor): ... ,
            cn_time_decay (bool): indicate whether applying decay on time,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        tar_i, tar_j = tar_ei[0], tar_ei[1]
        if cn_time_decay:
            assert time_info is not None
            last_update, pos_t = time_info
            last_update = last_update.unsqueeze(0)  # 1*N
            pos_t = pos_t.unsqueeze(1)  # B*1
            time_decay_matrix = (pos_t - last_update) / 10000  # time scale
            time_decay_matrix = torch.exp(-time_decay_matrix)

        id_num = x.size(0)

        if self.k == 4:
            adj0 = SparseTensor.eye(id_num, device=x.device)
            adj1 = (
                SparseTensor.from_edge_index(
                    torch.cat(
                        (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                        dim=-1,
                    ),
                    sparse_sizes=(id_num, id_num),
                )
                .fill_value_(1.0)
                .coalesce()
                .to(x.device)
            )
            i_0_v, i_1_v, j_0_v, j_1_v = (
                adj0[tar_i],
                adj1[tar_i],
                adj0[tar_j],
                adj1[tar_j],
            )
            # print(adj1)
            # print(tar_j)
            # print(j_1_v)
            # exit()
            # print(i_0_v)
            i_0_e, i_1_e, j_0_e, j_1_e = (
                i_0_v.fill_value_(1.0),
                i_1_v.fill_value_(1.0),
                j_0_v.fill_value_(1.0),
                j_1_v.fill_value_(1.0),
            )
            # print(i_0_e)
            # exit()

            # print("========")
            # print(i_0_v)
            # print(j_1_v)
            # print(i_0_v * j_1_v)
            # exit()
            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v

            if cn_time_decay:
                cn_0_1, cn_1_0, cn_1_1 = (
                    cn_0_1 * time_decay_matrix,
                    cn_1_0 * time_decay_matrix,
                    cn_1_1 * time_decay_matrix,
                )
            xcn_0_1, xcn_1_0, xcn_1_1 = (
                spmm_add(cn_0_1, x),
                spmm_add(cn_1_0, x),
                spmm_add(cn_1_1, x),
            )
            cn_emb = torch.cat([xcn_0_1, xcn_1_0, xcn_1_1], dim=-1)
            print("=======Sparse==========")
            print(cn_emb)

        elif self.k == 2:
            adj1 = (
                SparseTensor.from_edge_index(
                    torch.cat(
                        (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                        dim=-1,
                    ),
                    sparse_sizes=(id_num, id_num),
                )
                .fill_value_(1.0)
                .coalesce()
                .to(x.device)
            )
            i_1_v, j_1_v = adj1[tar_i], adj1[tar_j]
            i_1_e, j_1_e = i_1_v.fill_value_(1.0), j_1_v.fill_value_(1.0)
            cn_1_1 = i_1_v * j_1_v
            if cn_time_decay:
                cn_1_1 = cn_1_1 * time_decay_matrix
            xcn_1_1 = spmm_add(cn_1_1, x)
            cn_emb = torch.cat([xcn_1_1], dim=-1)

        elif self.k == 8:
            adj0 = SparseTensor.eye(id_num, device=x.device)
            adj1 = (
                SparseTensor.from_edge_index(
                    torch.cat(
                        (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                        dim=-1,
                    ),
                    sparse_sizes=(id_num, id_num),
                )
                .fill_value_(1.0)
                .coalesce()
                .to(x.device)
            )
            adj2 = adj1.matmul(adj1)  # self: fake 2 hop
            k3cycle = adj2.matmul(adj1)
            i_0_v, i_1_v, i_2_v, j_0_v, j_1_v, j_2_v = (
                adj0[tar_i],
                adj1[tar_i],
                adj2[tar_i],
                adj0[tar_j],
                adj1[tar_j],
                adj2[tar_j],
            )
            i_0_e, i_1_e, i_2_e, j_0_e, j_1_e, j_2_e = (
                i_0_v.fill_value_(1.0),
                i_1_v.fill_value_(1.0),
                i_2_v.fill_value_(1.0),
                j_0_v.fill_value_(1.0),
                j_1_v.fill_value_(1.0),
                j_2_v.fill_value_(1.0),
            )

            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v
            cn_1_2, cn_2_1, cn_2_2 = ((i_1_v * j_2_v), (i_2_v * j_1_v), (i_2_v * j_2_v))
            u_v_value = adj1[tar_i, tar_j].to_dense().diag().reshape(-1, 1) * (-1)
            delta_1_2 = i_1_v * i_1_v * u_v_value
            delta_2_1 = j_1_v * j_1_v * u_v_value
            row, col, value = cn_1_1.coo()
            neg_cn_1_1 = SparseTensor(
                row=row, col=col, value=-value, sparse_sizes=cn_1_1.sparse_sizes()
            ).to_device(x.device)
            delta_2_2 = (
                i_1_e * k3cycle[tar_i, tar_i].to_dense().diag().reshape(-1, 1)
                + j_1_e * k3cycle[tar_j, tar_j].to_dense().diag().reshape(-1, 1)
                + neg_cn_1_1
            ) * u_v_value
            special_2_2 = cn_1_1.matmul(adj1)
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

            if cn_time_decay:
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
                    SparseTensor.from_dense(cn_0_1),
                    SparseTensor.from_dense(cn_1_0),
                    SparseTensor.from_dense(cn_1_1),
                )
            cn_1_2, cn_2_1, cn_2_2 = (
                SparseTensor.from_dense(cn_1_2),
                SparseTensor.from_dense(cn_2_1),
                SparseTensor.from_dense(cn_2_2),
            )
            xcn_0_1, xcn_1_0, xcn_1_1, xcn_1_2, xcn_2_1, xcn_2_2 = (
                spmm_add(cn_0_1, x),
                spmm_add(cn_1_0, x),
                spmm_add(cn_1_1, x),
                spmm_add(cn_1_2, x),
                spmm_add(cn_2_1, x),
                spmm_add(cn_2_2, x),
            )
            special_xcn_2_2 = spmm_add(special_2_2, x)
            cn_emb = torch.cat(
                [xcn_0_1, xcn_1_0, xcn_1_1, xcn_1_2, xcn_2_1, xcn_2_2, special_xcn_2_2],
                dim=-1,
            )

        return cn_emb

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        cn_time_decay: bool = False,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edge list,
            tar_ei (torch.Tensor): ... ,
            cn_time_decay (bool): indicate whether applying decay on time,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]

        xij = torch.mul(xi, xj).reshape(-1, x.size(1))

        cn_emb = self.get_cn_emb(x, edge_index, tar_ei, cn_time_decay, time_info)
        xs = torch.cat([xij, cn_emb], dim=-1)

        xs.relu()
        xs = self.xsmlp(xs)

        return xs


class NCNPredictor(torch.nn.Module):
    r"""An implementation of Temporal Neural Common Neighbor (TNCN).

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_dim (int): Size of each hidden embedding.
        k (int): ...

    Reference: https://arxiv.org/abs/2406.07926.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        k: int = 2,
    ):
        super().__init__()
        assert k in [2, 4, 8], 'Please choose k values from [2,4,8]'

        self.k = k
        self.xslin = torch.nn.Linear(
            k * in_channels, out_channels
        )  # TODO: add more layers
        self.xsmlp = torch.nn.Sequential(
            torch.nn.Linear(k * in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_channels),
        )

    def get_cn_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        cn_time_decay: bool = False,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Obtain the CNs embeddings for each node pair in a batch.

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edge list,
            tar_ei (torch.Tensor): ... ,
            cn_time_decay (bool): indicate whether applying decay on time,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        tar_i, tar_j = tar_ei[0], tar_ei[1]
        if cn_time_decay:
            assert time_info is not None
            last_update, pos_t = time_info
            last_update = last_update.unsqueeze(0)  # 1*N
            pos_t = pos_t.unsqueeze(1)  # B*1
            time_decay_matrix = (pos_t - last_update) / 10000  # time scale
            time_decay_matrix = torch.exp(-time_decay_matrix)

        id_num = x.size(0)

        if self.k == 4:
            # adj0 = SparseTensor.eye(id_num, device=x.device)
            indices = torch.arange(id_num, device=x.device)
            adj0 = torch.sparse_coo_tensor(
                torch.stack([indices,indices],dim=0),
                torch.ones(id_num,device=x.device),
                size=(id_num,id_num),
                device=x.device
            )

            # adj1 = (
            #     SparseTensor.from_edge_index(
            #         torch.cat(
            #             (edge_index, torch.stack([edge_index[1], edge_index[0]])),
            #             dim=-1,
            #         ),
            #         sparse_sizes=(id_num, id_num),
            #     )
            #     .fill_value_(1.0)
            #     .coalesce()
            #     .to(x.device)
            # )
            adj1 = torch.sparse_coo_tensor(
                torch.cat(
                    (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                    dim=-1,
                ),
                torch.ones(edge_index.shape[1] * 2), size=(id_num, id_num)
            ).coalesce().to(x.device)

            # i_0_v, i_1_v, j_0_v, j_1_v = (
            #     adj0[tar_i],
            #     adj1[tar_i],
            #     adj0[tar_j],
            #     adj1[tar_j],
            # )
            _sparse_row_select(adj1,tar_j)
            i_0_v, i_1_v, j_0_v, j_1_v = (
                _sparse_row_select(adj0,tar_i),
                _sparse_row_select(adj1,tar_i),
                _sparse_row_select(adj0,tar_j),
                _sparse_row_select(adj1,tar_j),
            )

            # print(adj1)
            # print(tar_j)
            # print(j_1_v)
            # exit()

            # i_0_e, i_1_e, j_0_e, j_1_e = (
            #     i_0_v.fill_value_(1.0),
            #     i_1_v.fill_value_(1.0),
            #     j_0_v.fill_value_(1.0),
            #     j_1_v.fill_value_(1.0),
            # )
            # print(i_0_v)
            i_0_e, i_1_e, j_0_e, j_1_e = (
                _fill(i_0_v,1.0),
                _fill(i_1_v,1.0),
                _fill(j_0_v,1.0),
                _fill(j_1_v,1.0),
            )
            # print(i_0_e)
            # exit()
            # print("========")
            # print(i_0_v)
            # print(j_1_v)
            # print(i_0_v * j_1_v)
            # exit()
            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v

            
            if cn_time_decay:
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
            print("=======Torch==========")
            print(cn_emb)

        elif self.k == 2:
            adj1 = (
                SparseTensor.from_edge_index(
                    torch.cat(
                        (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                        dim=-1,
                    ),
                    sparse_sizes=(id_num, id_num),
                )
                .fill_value_(1.0)
                .coalesce()
                .to(x.device)
            )
            i_1_v, j_1_v = adj1[tar_i], adj1[tar_j]
            i_1_e, j_1_e = i_1_v.fill_value_(1.0), j_1_v.fill_value_(1.0)
            cn_1_1 = i_1_v * j_1_v
            if cn_time_decay:
                cn_1_1 = cn_1_1 * time_decay_matrix
            xcn_1_1 = spmm_add(cn_1_1, x)
            cn_emb = torch.cat([xcn_1_1], dim=-1)

        elif self.k == 8:
            adj0 = SparseTensor.eye(id_num, device=x.device)
            adj1 = (
                SparseTensor.from_edge_index(
                    torch.cat(
                        (edge_index, torch.stack([edge_index[1], edge_index[0]])),
                        dim=-1,
                    ),
                    sparse_sizes=(id_num, id_num),
                )
                .fill_value_(1.0)
                .coalesce()
                .to(x.device)
            )
            adj2 = adj1.matmul(adj1)  # self: fake 2 hop
            k3cycle = adj2.matmul(adj1)
            i_0_v, i_1_v, i_2_v, j_0_v, j_1_v, j_2_v = (
                adj0[tar_i],
                adj1[tar_i],
                adj2[tar_i],
                adj0[tar_j],
                adj1[tar_j],
                adj2[tar_j],
            )
            i_0_e, i_1_e, i_2_e, j_0_e, j_1_e, j_2_e = (
                i_0_v.fill_value_(1.0),
                i_1_v.fill_value_(1.0),
                i_2_v.fill_value_(1.0),
                j_0_v.fill_value_(1.0),
                j_1_v.fill_value_(1.0),
                j_2_v.fill_value_(1.0),
            )

            cn_0_1, cn_1_0 = (i_0_v * j_1_v), (i_1_v * j_0_v)
            cn_1_1 = i_1_v * j_1_v
            cn_1_2, cn_2_1, cn_2_2 = ((i_1_v * j_2_v), (i_2_v * j_1_v), (i_2_v * j_2_v))
            u_v_value = adj1[tar_i, tar_j].to_dense().diag().reshape(-1, 1) * (-1)
            delta_1_2 = i_1_v * i_1_v * u_v_value
            delta_2_1 = j_1_v * j_1_v * u_v_value
            row, col, value = cn_1_1.coo()
            neg_cn_1_1 = SparseTensor(
                row=row, col=col, value=-value, sparse_sizes=cn_1_1.sparse_sizes()
            ).to_device(x.device)
            delta_2_2 = (
                i_1_e * k3cycle[tar_i, tar_i].to_dense().diag().reshape(-1, 1)
                + j_1_e * k3cycle[tar_j, tar_j].to_dense().diag().reshape(-1, 1)
                + neg_cn_1_1
            ) * u_v_value
            special_2_2 = cn_1_1.matmul(adj1)
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

            if cn_time_decay:
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
                    SparseTensor.from_dense(cn_0_1),
                    SparseTensor.from_dense(cn_1_0),
                    SparseTensor.from_dense(cn_1_1),
                )
            cn_1_2, cn_2_1, cn_2_2 = (
                SparseTensor.from_dense(cn_1_2),
                SparseTensor.from_dense(cn_2_1),
                SparseTensor.from_dense(cn_2_2),
            )
            xcn_0_1, xcn_1_0, xcn_1_1, xcn_1_2, xcn_2_1, xcn_2_2 = (
                spmm_add(cn_0_1, x),
                spmm_add(cn_1_0, x),
                spmm_add(cn_1_1, x),
                spmm_add(cn_1_2, x),
                spmm_add(cn_2_1, x),
                spmm_add(cn_2_2, x),
            )
            special_xcn_2_2 = spmm_add(special_2_2, x)
            cn_emb = torch.cat(
                [xcn_0_1, xcn_1_0, xcn_1_1, xcn_1_2, xcn_2_1, xcn_2_2, special_xcn_2_2],
                dim=-1,
            )

        return cn_emb

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tar_ei: torch.Tensor,
        cn_time_decay: bool = False,
        time_info: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): node features,
            edge_index (torch.Tensor): edge list,
            tar_ei (torch.Tensor): ... ,
            cn_time_decay (bool): indicate whether applying decay on time,
            time_info (Optional[Tuple[torch.Tensor, torch.Tensor]]): A tuple of last update and current time of each edge
        """
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]

        xij = torch.mul(xi, xj).reshape(-1, x.size(1))

        cn_emb = self.get_cn_emb(x, edge_index, tar_ei, cn_time_decay, time_info)
        xs = torch.cat([xij, cn_emb], dim=-1)

        xs.relu()
        xs = self.xsmlp(xs)

        return xs


if __name__ == '__main__':
    z = torch.randn(5, 5)  # [num_nodes, hidden_dim]
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],  # source nodes
            [1, 2, 3, 4],  # target nodes
        ],
        dtype=torch.long,
    )
    number_nodes = 6
    x = torch.rand(6, 2)

   

    # adj1 = (
    #     SparseTensor.from_edge_index(
    #         torch.cat(
    #             (edge_index, torch.stack([edge_index[1], edge_index[0]])),
    #             dim=-1,
    #         ),
    #         sparse_sizes=(number_nodes, number_nodes),
    #     )
    #     .fill_value_(1.0)
    #     .coalesce()
    # )
    # print(adj1.to_dense())

    # print(adj1)

    # two_way = torch.cat(
    #     (edge_index, torch.stack([edge_index[1], edge_index[0]])),
    #     dim=-1,
    # )
    # adj = torch.sparse_coo_tensor(
    #     two_way, torch.ones(two_way.shape[1]), size=(number_nodes, number_nodes)
    # )
    # print(adj.to_dense())

    # print(torch.equal(adj1.to_dense(), adj.to_dense()))  # True

    # # ======================
    # print(spmm_add(adj1, x))
    # print(torch.sparse.mm(adj, x))
    # print(torch.equal(spmm_add(adj1, x), torch.sparse.mm(adj, x)))  # True


    # #================
    # print("========")
    # adj0 = SparseTensor.eye(6, device=x.device)
    # indices = torch.arange(6, device=x.device)
    # adj1 = torch.sparse_coo_tensor(
    #     torch.stack([indices,indices],dim=0),
    #     torch.ones(6,device=x.device),
    #     size=(6,6),
    #     device=x.device
    # )
    # print(torch.equal(adj0.to_dense(), adj1.to_dense()))  # True
    # print("========")


    src = torch.tensor([0, 2], dtype=torch.long)
    dst = torch.tensor([3, 4], dtype=torch.long)

    model = NCNPredictor(in_channels=5, hidden_dim=5, out_channels=5, k=4)
    out1 = model(z, edge_index, edge_index)
    print(out1.shape)

    model = NCNPredictor_Sparse(in_channels=5, hidden_dim=5, out_channels=5, k=4)
    out2 = model(z, edge_index, edge_index)
    print(out2.shape)
    # print(torch.allclose(out1, out2, rtol=1e-4, atol=1e-5))

