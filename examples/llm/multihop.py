"""TGTalker multi-hop variant.

Same zero-shot link-prediction pipeline as ``TGTalker.py``, but the prompt
includes a multi-hop temporal neighborhood of the source node rather than just
its 1-hop history. Multi-hop neighbors are produced natively by
``RecencyNeighborHook`` via ``num_nbrs=[k] * hops``: ``batch.nbr_nids[h]`` holds
the hop-``h`` neighbors, where source ``i`` occupies the row range
``[i * k**h, (i + 1) * k**h)``.

See README.md for details.
"""

import argparse
import json
import logging

import numpy as np
import outlines
import torch
from schemas import TGAnswer, TGReasoning
from tgb.linkproppred.evaluate import Evaluator
from tgtalker_utils import (
    gather_hop_edges,
    make_multihop_user_prompt,
    make_system_prompt,
    predict_link,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tgm import DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry
from tgm.util.logging import enable_logging, log_metric
from tgm.util.seed import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='TGTalker Multi-hop LinkPropPred Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--dataset', type=str, default='tgbl-wiki', help='TGB dataset name'
    )
    parser.add_argument('--bsize', type=int, default=200, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-1.7B',
        help='Full huggingface model path',
    )
    parser.add_argument(
        '--n-nbrs', type=int, default=2, help='num recent neighbors per hop'
    )
    parser.add_argument('--hops', type=int, default=2, help='number of hops')
    parser.add_argument('--cot', action='store_true', help='enable chain-of-thought')
    parser.add_argument(
        '--max-test-edges',
        type=int,
        default=None,
        help='optional cap on number of test edges (useful for smoke tests)',
    )
    parser.add_argument('--log-file-path', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_logging(log_file_path=args.log_file_path)
    seed_everything(args.seed)

    logger.info(f'Loading dataset {args.dataset}...')
    full_data = DGData.from_tgb(args.dataset)
    train_data, val_data, test_data = full_data.split()
    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.n_nbrs] * args.hops,
        num_nodes=full_data.num_nodes,
        seed_nodes_keys=['edge_src'],
        seed_times_keys=['edge_time'],
        directed=True,
    )
    hm = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
    )
    train_key, val_key, test_key = hm.keys
    hm.register_shared(nbr_hook)

    logger.info(f'Loading model {args.model}...')
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = outlines.from_transformers(model, tokenizer)
    schema = TGReasoning if args.cot else TGAnswer

    evaluator = Evaluator(name=args.dataset)
    perf_list = []

    train_loader = DGDataLoader(train_dg, batch_size=200, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, batch_size=200, hook_manager=hm)
    logger.info('Warming up recency neighbor buffers on train/val...')
    with hm.activate(train_key):
        for _ in train_loader:
            pass
    with hm.activate(val_key):
        for _ in val_loader:
            pass

    logger.info('Starting inference on test set...')
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)
    system_prompt = make_system_prompt(use_cot=args.cot)

    n_seen = 0
    n_failed = 0
    stop = False
    with hm.activate(test_key):
        for batch in tqdm(test_loader, desc='Test Inference'):
            if stop:
                break
            srcs = batch.edge_src.cpu().numpy()
            times = batch.edge_time.cpu().numpy()
            seed_nids = [s.cpu().numpy() for s in batch.seed_nids]
            nbr_nids = [n.cpu().numpy() for n in batch.nbr_nids]
            nbr_times = [t.cpu().numpy() for t in batch.nbr_edge_time]

            for i in range(len(srcs)):
                hop_edges = gather_hop_edges(
                    i, args.hops, args.n_nbrs, seed_nids, nbr_nids, nbr_times
                )
                user_prompt = make_multihop_user_prompt(srcs[i], times[i], hop_edges)
                prompt = tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                try:
                    output = model(prompt, schema)
                    pred_dst = int(json.loads(output)['destination_node'])
                    query_dst = torch.cat(
                        [batch.edge_dst[i].unsqueeze(0), batch.neg_batch_list[i]]
                    )
                    y_pred = predict_link(query_dst, pred_dst)
                    input_dict = {
                        'y_pred_pos': y_pred[0],
                        'y_pred_neg': y_pred[1:],
                        'eval_metric': [METRIC_TGB_LINKPROPPRED],
                    }
                    perf_list.append(
                        evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED]
                    )
                except Exception as e:
                    logger.error(f'Generation/scoring failed: {e}')
                    perf_list.append(0.0)
                    n_failed += 1

                n_seen += 1
                if args.max_test_edges is not None and n_seen >= args.max_test_edges:
                    stop = True
                    break

    score = float(np.mean(perf_list)) if perf_list else 0.0
    logger.info(
        f'Test {METRIC_TGB_LINKPROPPRED.upper()} ({args.hops}-hop): {score:.4f} '
        f'over {len(perf_list)} edges ({n_failed} generation failures)'
    )
    log_metric(METRIC_TGB_LINKPROPPRED.upper(), score)


if __name__ == '__main__':
    main()
