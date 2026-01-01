import torch
from tgb.linkproppred.dataset import LinkPropPredDataset

# data loading with numpy
dataset = LinkPropPredDataset(name='tkgl-smallpedia', root='datasets', preprocess=True)
data = dataset.full_data
# print(data.node_type)

# print(type(LinkPropPredDataset[]))
print(torch.unique(torch.from_numpy(data['edge_feat']))[0])

print(dataset.edge_type.shape)
for key, value in data.items():
    print(f'{key}: {value.shape}')
