class BadHookProtocolError(Exception):
    """Raised when a DGHook does not correctly implement the required protocol for execution by the HookManager."""


class UnresolvableHookDependenciesError(Exception):
    """Raised when no valid execution ordering of hooks can be found, due to conflicting or cyclic requires/produces dependencies."""


class PaddedNodeIDError(Exception):
    """Raised when a dataset contains node IDs that conflict with the reserved padded node placeholder ID."""


class EmptyGraphError(Exception):
    """Raised when attempting to instantiate an empty graph. Empty graphs are unsupported since updates are not allowed."""


class OrderedTimeGranularityError(Exception):
    """Raised when an operation requiring time-based granularity is attempted on a graph that only has ordered (relative) granularity.

    Examples:
        - Discretizing or coarsening an ordered graph by absolute time units.
        - Iterating an ordered graph with non-ordered batch units (e.g. daily snapshots).
    """


class InvalidDiscretizationError(Exception):
    """Raised when attempting to discretize a graph to a finer granularity, which is undefined."""


class EmptyBatchError(Exception):
    """Raised during time-based iteration when a batch interval contains no events."""
