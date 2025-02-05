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
