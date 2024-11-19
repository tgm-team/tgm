"""
Script for testing data loading from csv 
$ pytest test_loading.py
"""
import csv
import torch
from opendg.data.storage import EventStore
from opendg.data.data import CTDG,DTDG
import os

def test_loading_EventStore():
    r"""Test the loading of a synthetic event store from a csv file."""
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
        event = EventStore(timestamp=t,edges=edge_index,node_feats=node_feats)
        assert event.num_edges == 1
        assert event.num_updated_nodes == 2
        assert event.edges.shape == edge_index.shape
        assert event.timestamp == t
        assert len(event.node_feats) == len(node_feats)

        #! test for device, not sure we want to include it as it requires cuda
        # event = event.to('cpu')
        # event = event.cpu()
        # assert event.device == 'cpu'
        # event = event.to('cuda')
        # event = event.cuda()
        # assert event.device == 'cuda'


    os.remove('test_ctdg.csv')


    
def test_loading_CTDG():
    r"""Test the loading of a CTDG from a csv file."""
    test_edges = [] # [(u,v,t)]
    test_size = 100
    fname = 'test_ctdg.csv'
    for i in range(0,test_size):
        test_edges.append((i,i+1,i))

    with open(fname, mode='w') as ctdg_file:
        ctdg_writer = csv.writer(ctdg_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ctdg_writer.writerow(['source','destination','timestamp'])
        for edge in test_edges:
            ctdg_writer.writerow(edge)

    #* test loading directly from csv
    tg = CTDG(fname)
    tg.load_csv(fname)
    assert tg.num_edges == test_size

    #! debug from here, why is it 200 instead of 100

            



# def test_loading_DTDG():
#     pass