import numpy as np
import torch
from relbench.datasets import get_dataset, get_dataset_names

schema_hm = {  # We need to maintain this ourselves. If RelBench changes it, we need to udpate this
    'article': {
        'article_id': 'numerical',
        'product_code': 'numerical',
        'prod_name': 'text',
        'product_type_no': 'numerical',
        'product_type_name': 'categorical',
        'product_group_name': 'categorical',
        'graphical_appearance_no': 'categorical',
        'graphical_appearance_name': 'categorical',
        'colour_group_code': 'categorical',
        'colour_group_name': 'categorical',
        'perceived_colour_value_id': 'categorical',
        'perceived_colour_value_name': 'categorical',
        'perceived_colour_master_id': 'numerical',
        'perceived_colour_master_name': 'categorical',
        'department_no': 'numerical',
        'department_name': 'categorical',
        'index_code': 'categorical',
        'index_name': 'categorical',
        'index_group_no': 'categorical',
        'index_group_name': 'categorical',
        'section_no': 'numerical',
        'section_name': 'text',
        'garment_group_no': 'categorical',
        'garment_group_name': 'categorical',
        'detail_desc': 'text',
    },
    'customer': {
        'customer_id': 'text',
        'FN': 'categorical',
        'Active': 'categorical',
        'club_member_status': 'categorical',
        'fashion_news_frequency': 'categorical',
        'age': 'numerical',
        'postal_code': 'categorical',
    },
    'transactions': {
        't_dat': 'timestamp',
        'price': 'numerical',
        'sales_channel_id': 'categorical',
        'customer_id': 'numerical',
        'article_id': 'numerical',
    },
}


# All available datasets
dataset_list = get_dataset_names()
print(dataset_list)


def data_loader(dataset_name='rel-hm'):
    dataset = get_dataset(name=dataset_name, download=True)
    db = dataset.get_db()
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


data_loader()
