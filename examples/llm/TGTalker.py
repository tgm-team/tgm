import argparse
import logging
from pydantic import BaseModel
import numpy as np
import torch
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM

from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm
import json


from tgm import DGraph
from tgm.constants import PADDED_NODE_ID, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry
from tgm.util.logging import enable_logging, log_metric
from tgm.util.seed import seed_everything
from tgm.constants import METRIC_TGB_LINKPROPPRED

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_link(query_dst: torch.Tensor, llm_dst: int) -> torch.Tensor:
    return (query_dst == llm_dst).float()

class Step(BaseModel):
    explanation: str
    output: str

class TGAnswer(BaseModel):
    destination_node: int

def make_user_prompt(src, ts, nbr_nids=None, nbr_times=None):
    if nbr_nids is not None and len(nbr_nids) > 0:
        user_prompt = f',Source Node` {src} has the following past interactions:\n'
        # Filter out padded nodes
        valid_indices = [i for i, n in enumerate(nbr_nids) if n != PADDED_NODE_ID]

        # Sort by time if needed, but RecencyNeighborHook usually gives most recent.
        # reasoning_main.py iterates and prints.
        # TGM RecencyNeighborHook returns neighbors.
        # Assuming nbr_nids and nbr_times are lists/arrays for this specific source.

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


def main():
    parser = argparse.ArgumentParser(description='TGTalker LinkPropPred Example')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
    parser.add_argument(
        '--bsize', type=int, default=200, help='batch size (default 200 for TGB)'
    )
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-1.7B',
        help='Full huggingface model path',
    )
    parser.add_argument(
        '--n-nbrs', type=int, default=10, help='num sampled nbrs (recency)'
    )
    parser.add_argument('--log-file-path', type=str, default=None)

    args = parser.parse_args()
    enable_logging(log_file_path=args.log_file_path)
    seed_everything(args.seed)

    logger.info(f'Loading dataset {args.dataset}...')
    # Load dataset
    # We only need test data for TGTalker typically, but RecencyHook needs state from Train/Val usually?
    # reasoning_main.py loads everything.
    # In TGM, RecencyNeighborHook needs to be updated with history.
    # So we should probably run through train/val to populate the hook, or just load full data.
    # DGData.split() gives train/val/test.
    train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

    # We need to populate the neighbor hook with historical data up to the test point.
    # So we combine train and val for the 'historical' graph construction if we want the hook to have info.
    # Or we can just use the provided split standard.
    # TGM's RecencyNeighborHook updates its state as it processes batches.
    # To have valid history for test set, we generally need to replay train/val.

    logger.info('Initializing Graphs...')
    # Construct graphs
    # Note: For RecencyHook to work on test_loader, it usually needs to have seen previous edges.
    # However, if we just want to run inference on Test, we might strictly need the history available.
    # TGM loaders usually handle this by stateful hooks.
    # We will assume we replay train/val to build state, or if the dataset is small enough.
    # For efficiency in this example code, we might just instantiate the hook and if possible pre-load it?
    # RecencyNeighborHook in TGM is updated via `after_batch` or implicit updates.

    # Let's perform a quick "warmup" pass if possible, or just run normally.
    # Since this is an example, we will follow standard tgm example pattern:
    # Build hooks, register them.

    train_dg = DGraph(train_data)
    val_dg = DGraph(val_data)
    test_dg = DGraph(test_data)

    # Setup Hook
    # We use num_nodes from the full dataset or max id.
    # Assuming test_dg has max node id.
    max_node_id = max(train_dg.num_nodes, val_dg.num_nodes, test_dg.num_nodes)

    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],  # One hop
        num_nodes=max_node_id + 1000,  # Safety buffer
        seed_nodes_keys=['src'],  # We only care about src history for the prompt
        seed_times_keys=['time'],
    )

    hm = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
    )
    # We register our specific hook.
    # Note: The recipe might already have a neighbor sampler. We should check if we are replacing or adding.
    # RECIPE_TGB_LINK_PRED typically has a sampling strategy.
    # But here we want OUR recency hook to be the one providing data for the prompt.
    # We can just use the hook standalone or register it.
    hm.register_shared(nbr_hook)

    train_key, val_key, test_key = hm.keys

    # Prepare Model
    logger.info(f'Loading Model {args.model}...')
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = outlines.from_transformers(model, tokenizer)
        
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        return

    evaluator = Evaluator(name=args.dataset)
    perf_list = []

    # WARMUP: We need to push Train and Val edges into the RecencyHook so Test set lookup finds them.
    # TGM hooks update on `after_batch`.
    # We can run a fast pass over train/val without LLM inference just to update the hook.

    logger.info('Warming up RecencyHook with Train/Val data...')
    # Helper to push edges to hook without inference
    # RecencyNeighborHook updates internal state based on the batch data it sees.
    # We can just iterate loaders.

    train_loader = DGDataLoader(
        train_dg, batch_size=200, hook_manager=hm
    )  # Larger batch for fast warmup
    val_loader = DGDataLoader(val_dg, batch_size=200, hook_manager=hm)

    with hm.activate(train_key):
        for batch in train_loader:
            pass  # Hook automatically updates state
        print ("setting neighbor states for training set")

    with hm.activate(val_key):
        for batch in val_loader:
            pass
        print ("setting neighbor states for validation set")

    # INFERENCE ON TEST
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

    logger.info('Starting Inference on Test set...')

    count = 0
    with hm.activate(test_key):
        for batch in tqdm(test_loader, desc='Test Inference'):
            # batch.nids[0] keys: 'src', 'dst', 'neg' usually if link pred
            # src nodes: batch.src
            # times: batch.times[0] (seed times)

            # The nbr_hook should have populated batch with neighbors.
            # batch.nbr_nids -> List[Tensor]. batch.nbr_nids[0] for first hop.
            # shape: [batch_size, num_nbrs]

            srcs = batch.src.cpu().numpy()
            times = batch.times[0].cpu().numpy()

            if hasattr(batch, 'nbr_nids') and len(batch.nbr_nids) > 0:
                nbr_nids_batch = batch.nbr_nids[0].cpu().numpy()
                nbr_times_batch = batch.nbr_times[0].cpu().numpy()
            else:
                # Should not happen if hook is working
                nbr_nids_batch = np.full((len(srcs), args.n_nbrs), PADDED_NODE_ID)
                nbr_times_batch = np.zeros((len(srcs), args.n_nbrs))

            # Iterate through batch
            for i in range(len(srcs)):
                src = srcs[i]
                ts = times[i]
                current_nbr_nids = nbr_nids_batch[i]
                current_nbr_times = nbr_times_batch[i]

                system_prompt = make_system_prompt()
                user_prompt = make_user_prompt(
                    src, ts, current_nbr_nids, current_nbr_times
                )

                # Construct full prompt for Chat model
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                try:
                    output = model(
                            prompt,
                            TGAnswer,
                        )
                    data = json.loads(output)
                    pred_dst = int(data['destination_node'])
                    query_dst = torch.cat([batch.dst[i].unsqueeze(0), batch.neg_batch_list[i]])
                    y_pred = predict_link(query_dst, pred_dst)

                    input_dict = {
                        'y_pred_pos': y_pred[0],
                        'y_pred_neg': y_pred[1:],
                        'eval_metric': [METRIC_TGB_LINKPROPPRED],
                    }
                    perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])
                except Exception as e:
                    logger.error(f'Generation failed: {e}')
                    perf_list.append(0.0)
                count += 1

    score = np.mean(perf_list) if perf_list else 0.0
    logger.info(f'Test Score (MRR): {score}')
    log_metric('MRR', score)


if __name__ == '__main__':
    main()
