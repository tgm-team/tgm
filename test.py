import torch

nodes_type = torch.tensor([0, 1, 2, 1, 0, 1])

dst_node_types = torch.tensor([1, 0, 3])


match_matrix = dst_node_types.unsqueeze(1) == nodes_type.unsqueeze(0)
valid_neg_mask = match_matrix.any(dim=1)
rand_matrix = torch.where(
    match_matrix,
    torch.rand_like(match_matrix, dtype=torch.float),
    torch.tensor(-float('inf')),
)

selected = rand_matrix.argmax(dim=1)
selected[~valid_neg_mask] = -1

print(selected)
