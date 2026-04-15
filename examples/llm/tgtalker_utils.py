from tgm.constants import PADDED_NODE_ID


def make_user_prompt(src: int, ts: int, nbr_nids=None, nbr_times=None):
    if src is None or ts is None:
        raise ValueError('Source node and timestamp must be provided')

    if not isinstance(src, int) or not isinstance(ts, int):
        raise ValueError('Source node and timestamp must be integers')

    if nbr_nids is not None and len(nbr_nids) > 0:
        user_prompt = f',Source Node` {src} has the following past interactions:\n'
        # Filter out padded nodes
        valid_indices = [i for i, n in enumerate(nbr_nids) if n != PADDED_NODE_ID]

        for idx in valid_indices:
            dst = int(nbr_nids[idx])
            timestamp = int(nbr_times[idx])
            user_prompt += f'{src}, {dst}, {timestamp}) \n'

        user_prompt += f'Please predict the most likely `Destination Node` for `Source Node` {src} at `Timestamp` {ts}.'
    else:
        user_prompt = (
            f'Predict the next interaction for source node {src} at time {ts},'
        )
    return user_prompt


def make_system_prompt():
    system_prompt = (
        f'You are an expert temporal graph learning agent. Your task is to predict the next interaction (i.e. Destination Node) given the `Source Node` and `Timestamp`.\n\n'
        f'Description of the temporal graph is provided below, where each line is a tuple of (`Source Node`, `Destination Node`, `Timestamp`).\n\nTEMPORAL GRAPH:\n'
    )
    return system_prompt
