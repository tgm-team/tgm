"""
Script for testing data loading from csv 
$ pytest test_loading.py
"""
import csv
import torch
from opendg.data.storage import EventStore


def test_loading_EventStore():
    r"""Test the loading of a synthetic CTDG dataset from a csv file."""
    test_edges = [] # [(u,v,t)]
    test_size = 1
    for i in range(0,test_size):
        test_edges.append((i,i+1,i))
    
    with open('test_ctdg.csv', mode='w') as ctdg_file:
        ctdg_writer = csv.writer(ctdg_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ctdg_writer.writerow(['source','destination','timestamp'])
        for edge in test_edges:
            ctdg_writer.writerow(edge)

    with open('test_ctdg.csv', mode='r') as ctdg_file:
        ctdg_reader = csv.reader(ctdg_file)
        header = next(ctdg_reader)
        edge = next(ctdg_reader)
        u = int(edge[0])
        v = int(edge[1])
        t = int(edge[2])
    
    edge_index = torch.tensor([[u],[v]])
    node_feats = {u:torch.tensor([1]),v:torch.tensor([2])}  
    events = EventStore(timestamp=t,edges=edge_index,node_feats=node_feats)
    assert events.num_edges == 1
    assert events.num_updated_nodes == 2

    
# def test_loading_CTDG():
#     pass


# def test_loading_DTDG():
#     pass