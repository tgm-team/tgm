import numpy as np
import torch
from relbench.datasets import get_dataset, get_dataset_names

from relbench.modeling.utils import get_stype_proposal

from relbench.tasks import get_task_names, get_task

from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input



# All available datasets
dataset_list = get_dataset_names()
print(dataset_list)


def data_loader(dataset_name='rel-hm'):
    dataset = get_dataset(name=dataset_name, download=True)
    db = dataset.get_db()


    col_to_stype_dict = get_stype_proposal(db)
    print(col_to_stype_dict)
    print(col_to_stype_dict.keys())
    exit()
    all_schema = db.table_dict.keys()
    print(all_schema)

    table_article = db.table_dict['article'].df
    table_customer = db.table_dict['customer'].df
    table_transactions = db.table_dict['transactions'].df

    print(table_article.head(5))
    print(table_customer.head(5))
    print(table_transactions.head(5))

    """
    Transaction table is the edgelist and edge feature

    All article_id from article and customer_id from customer starts from 0
    We need to make these ID continuously align with  each other:
        article_id start from 0 to n; customer_id start from n to m
        where n is the number of article and m is the numer of customer
    """
    max_article_id = table_article['article_id'].max()
    print(max_article_id)

    table_customer['customer_id'] = table_customer['customer_id'] + max_article_id + 1
    print(table_customer['customer_id'].max())
    table_transactions['customer_id'] = (
        table_transactions['customer_id'] + max_article_id + 1
    )

    print('==' * 60)
    """
    Construct edge index and edge feature
    """
    table_transactions['t_dat'] = table_transactions['t_dat'].astype(
        np.int64
    )  # Conver to unix timestamp
    time = torch.from_numpy(table_transactions['t_dat'].to_numpy())
    edge_type = torch.zeros_like(time)  # rel-hm only has 1 "fake" table

    # @TODO: Confirm to see if this is true for other rel datasets:
    # For fake table, we skip the first 3 columns for time, src and dst. Take the rest for edge feature
    edge_feat = torch.from_numpy(table_transactions.iloc[:, 3:].to_numpy())

    print('edge type:', edge_type.shape)
    print('time:', time.shape)
    print('edge feature:', edge_feat.shape)

    print('==' * 60)
    """
    Article node features
    """
    article_node_types = torch.zeros(max_article_id)
    print('article node type:', article_node_types.shape)

    for col in table_article.columns[1:]:
        print(table_article[col].value_counts())

    print('==' * 60)
    """
    Customer node feature
    """

    customer_node_types = torch.ones(table_transactions['customer_id'].max())
    print('customer node type:', customer_node_types.shape)

    print('==' * 60)
    """
    Node feature
    """
    node_type = torch.cat([article_node_types, customer_node_types])
    print('node type:', node_type.shape)

    print('==' * 60)



from typing import List, Optional
from sentence_transformers import SentenceTransformer
from torch import Tensor
import os

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))

def load_data_from_relbench():
    dataset = get_dataset("rel-hm", download=True)
    task = get_task("rel-hm", 'user-churn', download=True)

    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    col_to_stype_dict

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    table_input = get_node_train_table_input(
        table=train_table,
        task=task,
    )

    print(table_input)

    # print(test_table)
    exit()

    text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device='cpu'), batch_size=256)

    for table_name, table in db.table_dict.items():
        df = table.df
        print(table_name)
        print(df)

    # data, col_stats_dict = make_pkey_fkey_graph(
    #     db,
    #     col_to_stype_dict=col_to_stype_dict,  # speficied column types
    #     text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
    #     cache_dir=os.path.join(
    #         './', f"rel-f1_materialized_cache"
    #     ),  # store materialized graph for convenience
    # )

    # print(data['transactions']['tf'])
    # # print(data['transactions']['tf'].feat_dict['sales_channel_id'])
    # tf = data['transactions']['tf']
    # for stype, tensor in tf.feat_dict.items():
    #     print(f"--- {stype} ---")
    #     print(tf.col_names_dict[stype])  # column names
    #     print(tensor)

load_data_from_relbench()

    

