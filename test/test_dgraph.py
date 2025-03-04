from opendg.graph import DGraph
from opendg.timedelta import TimeDeltaDG


def test_init_empty_dgraph_no_time_delta():
    events = []
    dgraph = DGraph(events)
    assert dgraph.time_delta.is_ordered


def test_init_empty_dgraph_with_time_delta():
    events = []
    td = TimeDeltaDG(unit='ms', value=5)
    dgraph = DGraph(events, td)
    assert dgraph.time_delta == td


"""
test init (with/without time delta)
test init empty (with/without time delta)
test init from events (with/without time delta)
test init from storage (with/without time delta)
test init bad storage (with/without time delta)

test to/from csv
test to/from csv with materialize

test slice time * ensure storage not touched and old is not messed up
test slice time bad args * ensure storage not touched and old is not messed up
test slice time force cache refresh * ensure storage not touched and old is new messed up
test slice time to empty * ensure storage not touched and old is not messed up
test slice time with shallow copy change * ensure storage not touched and old is not messed up
test slice time bad args with shallow copy change * ensure storage not touched and old is not messed up
test slice time force cache refresh with shallow copy change * ensure storage not touched and old is new messed up
test slice time to empty with shallow copy change * ensure storage not touched and old is not messed up

test slice nodes * ensure storage not touched and old is not messed up
test slice nodes force cache refresh * ensure storage not touched and old is not messed up
test slice nodes to empty * ensure storage not touched and old is not messed up
test slice nodes with shallow copy change * ensure storage not touched and old is not messed up
test slice nodes force cache refresh with shallow copy change* ensure storage not touched and old is not messed up
test slice nodes to empty with shallow copy change* ensure storage not touched and old is not messed up

test multi slice time/nodes * ensure storage not touched and old is not messed up

test append single event  * check copy on write / materialize
test append multiple events * check copy on write / materilize
test append bad event args * check copy on write / materiliaze
test append single event with shallow copy change * check copy on write / materialize
test append multiple events with shallow copy change * check copy on write / materilize
test append bad event args with shallow copy change * check copy on write / materiliaze

test temporal coarsening sum good args with no features
test temporal coarsening sum with shallow copy change bad with no features
test temporal coarsening sum good args with features
test temporal coarsening sum with shallow copy change bad with features
test temporal coarsening concat good args with no features
test temporal coarsening concat with shallow copy change bad with no features
test temporal coarsening concat good args with features
test temporal coarsening concat with shallow copy change bad with features
test temporal coarsening empty graph
test temporal coarsening bad time delta
test temporal coarsening bad agg func
test temporal coarsening causes time gap

# And check these properties on each test case
test len
test start_time
test end_time
test time_delta
test num_nodes
test num_edges
test num_timestamps
test node_feats
test edge_feats
"""
