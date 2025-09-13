class TGMError(Exception):
    """Base class for all TGM library errors."""


class BadHookProtocolError(TGMError):
    """Raised when a DGHook does not correctly implement the required protocol for execution by the HookManager."""


class UnresolvableHookDependenciesError(TGMError):
    """Raised when no valid execution ordering of hooks can be found, due to conflicting or cyclic requires/produces dependencies."""


class InvalidNodeIDError(TGMError):
    """Raised when a dataset contains node IDs that conflict with the reserved padded node placeholder ID (tgm.constants.PADDED_NODE_ID)."""


class EmptyGraphError(TGMError):
    """Raised when attempting to instantiate an empty graph. Empty graphs are unsupported since updates are not allowed."""


class EventOrderedConversionError(TGMError):
    """Raised when an operation requiring time-ordered granularity is attempted on a graph that only has event-ordered granularity.

    Examples:
        - Discretizing or coarsening an event-ordered graph by absolute time units.
        - Iterating an even-ordered graph with time-ordered batch units (e.g. daily snapshots).
    """


class InvalidDiscretizationError(TGMError):
    """Raised when attempting to discretize a graph to a finer granularity, which is undefined."""


class EmptyBatchError(TGMError):
    """Raised during time-based iteration when a batch interval contains no events."""


class UndefinedRecipe(TGMError):
    """Raised when attempting to construct a recipe that is not defined/registered."""
