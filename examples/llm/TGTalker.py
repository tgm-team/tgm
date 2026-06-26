"""TGTalker: zero-shot temporal link prediction with a frozen LLM.

Reference port of https://github.com/shenyangHuang/TGTalker (paper:
"Are Large Language Models Good Temporal Graph Learners?", arXiv:2506.05393)
onto tgm primitives.

A node's recent temporal-graph history is serialized into a text prompt and a
frozen Hugging Face causal LM is asked to predict the next `Destination Node`.
Predictions are scored against TGB negatives with MRR. Supports the base mode
plus three optional enhancements from the paper:

  * a global background-edge context window  (``--bg-size``)
  * in-context-learning demonstrations       (``--icl`` + ``--in-size``)
  * chain-of-thought reasoning               (``--cot``)

See README.md for the full method description and how it maps onto tgm.
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
    BackgroundBuffer,
    ICLWindow,
    make_system_prompt,
    make_user_prompt,
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
        description='TGTalker LinkPropPred Example',
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
        '--n-nbrs',
        type=int,
        default=2,
        help='num recent neighbors of the source node to include (orig --nbr)',
    )
    parser.add_argument(
        '--bg-size',
        type=int,
        default=300,
        help='size of the global background-edge context window (orig --bg_size)',
    )
    parser.add_argument(
        '--in-size',
        type=int,
        default=5,
        help='number of in-context-learning demonstrations (orig --in_size)',
    )
    parser.add_argument('--icl', action='store_true', help='enable in-context learning')
    parser.add_argument('--cot', action='store_true', help='enable chain-of-thought')
    parser.add_argument(
        '--max-test-edges',
        type=int,
        default=None,
        help='optional cap on number of test edges (useful for smoke tests)',
    )
    parser.add_argument('--log-file-path', type=str, default=None)
    return parser.parse_args()


def warmup_and_seed_context(
    loader: DGDataLoader,
    hm,
    key: str,
    bg: BackgroundBuffer,
    icl: ICLWindow,
) -> None:
    """Iterate a loader to populate the recency hook's buffers and (for the most
    recent edges) seed the background / ICL sliding windows.
    """
    with hm.activate(key):
        for batch in loader:
            bg.extend(
                batch.edge_src.tolist(),
                batch.edge_dst.tolist(),
                batch.edge_time.tolist(),
            )
            icl.extend(
                batch.edge_src.tolist(),
                batch.edge_dst.tolist(),
                batch.edge_time.tolist(),
            )


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

    # Recency neighbor sampling, seeded on the source node only -- we just need
    # each source's recent interaction history for the prompt.
    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],
        num_nodes=full_data.num_nodes,
        seed_nodes_keys=['edge_src'],
        seed_times_keys=['edge_time'],
        directed=True,  # outgoing src->dst history only, matching the original
    )
    hm = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
    )
    train_key, val_key, test_key = hm.keys
    hm.register_shared(nbr_hook)

    # Sliding-window context shared across queries within a batch.
    bg = BackgroundBuffer(max_size=args.bg_size)
    icl = ICLWindow(max_size=args.in_size if args.icl else 0)

    logger.info(f'Loading model {args.model}...')
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = outlines.from_transformers(model, tokenizer)
    schema = TGReasoning if args.cot else TGAnswer

    evaluator = Evaluator(name=args.dataset)
    perf_list = []

    # Warm up recency buffers on train + val, and seed the background/ICL
    # windows from the most recent (validation) edges. A larger batch keeps
    # warmup fast; chronological order is preserved by the event-ordered loader.
    train_loader = DGDataLoader(train_dg, batch_size=200, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, batch_size=200, hook_manager=hm)
    logger.info('Warming up recency neighbor buffers on train/val...')
    warmup_and_seed_context(train_loader, hm, train_key, bg, icl)
    warmup_and_seed_context(val_loader, hm, val_key, bg, icl)

    logger.info('Starting inference on test set...')
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

    n_seen = 0
    n_failed = 0
    stop = False
    with hm.activate(test_key):
        for batch in tqdm(test_loader, desc='Test Inference'):
            if stop:
                break
            srcs = batch.edge_src.cpu().numpy()
            dsts = batch.edge_dst.cpu().numpy()
            times = batch.edge_time.cpu().numpy()

            if not (hasattr(batch, 'nbr_nids') and len(batch.nbr_nids) > 0):
                raise ValueError('Neighbor hook did not populate batch neighbors')
            nbr_nids_batch = batch.nbr_nids[0].cpu().numpy()
            nbr_times_batch = batch.nbr_edge_time[0].cpu().numpy()

            # Context (background graph + ICL demos) is shared within the batch.
            system_prompt = make_system_prompt(
                background_rows=bg.rows(),
                demos=icl.demos() if args.icl else None,
                use_cot=args.cot,
            )

            for i in range(len(srcs)):
                user_prompt = make_user_prompt(
                    srcs[i], times[i], nbr_nids_batch[i], nbr_times_batch[i]
                )
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
                    # Candidates: true destination followed by TGB negatives.
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

            # Reveal this batch's edges to the context windows *after* predicting
            # them, so background/ICL context never leaks the current edge.
            bg.extend(srcs.tolist(), dsts.tolist(), times.tolist())
            icl.extend(srcs.tolist(), dsts.tolist(), times.tolist())

    score = float(np.mean(perf_list)) if perf_list else 0.0
    logger.info(
        f'Test {METRIC_TGB_LINKPROPPRED.upper()}: {score:.4f} over '
        f'{len(perf_list)} edges ({n_failed} generation failures)'
    )
    log_metric(METRIC_TGB_LINKPROPPRED.upper(), score)


if __name__ == '__main__':
    main()
