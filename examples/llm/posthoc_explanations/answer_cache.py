"""Stage 1 of the post-hoc explanation pipeline: cache base-model answers.

Runs the TGTalker base model over the test set and dumps, for each edge, the
prompt shown to the model, the predicted destination, the ground-truth
destination, and whether the prediction was correct. The cache (JSONL) is the
input to ``prompt_cache.py``.

This stage requires the ``llm`` extra (outlines + transformers).
"""

import argparse
import json
import logging
import os
import sys

import outlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tgm import DGraph
from tgm.constants import RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry
from tgm.util.seed import seed_everything

# Allow running this script directly from the posthoc_explanations/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import TGAnswer  # noqa: E402
from tgtalker_utils import (  # noqa: E402
    BackgroundBuffer,
    make_system_prompt,
    make_user_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TGTalker answer cache')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='tgbl-wiki')
    parser.add_argument('--bsize', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--n-nbrs', type=int, default=2)
    parser.add_argument('--bg-size', type=int, default=300)
    parser.add_argument(
        '--max-test-edges',
        type=int,
        default=5000,
        help='cap on cached edges (the original caches the first 5,000)',
    )
    parser.add_argument(
        '--out', type=str, default='answer_cache/answers.jsonl', help='output JSONL'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    full_data = DGData.from_tgb(args.dataset)
    train_data, val_data, test_data = full_data.split()
    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],
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

    bg = BackgroundBuffer(max_size=args.bg_size)

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = outlines.from_transformers(model, tokenizer)

    train_loader = DGDataLoader(train_dg, batch_size=200, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, batch_size=200, hook_manager=hm)
    with hm.activate(train_key):
        for _ in train_loader:
            pass
    with hm.activate(val_key):
        for batch in val_loader:
            bg.extend(
                batch.edge_src.tolist(),
                batch.edge_dst.tolist(),
                batch.edge_time.tolist(),
            )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

    n = 0
    stop = False
    with open(args.out, 'w') as f, hm.activate(test_key):
        for batch in tqdm(test_loader, desc='Caching answers'):
            if stop:
                break
            srcs = batch.edge_src.cpu().numpy()
            dsts = batch.edge_dst.cpu().numpy()
            times = batch.edge_time.cpu().numpy()
            nbr_nids_batch = batch.nbr_nids[0].cpu().numpy()
            nbr_times_batch = batch.nbr_edge_time[0].cpu().numpy()
            system_prompt = make_system_prompt(background_rows=bg.rows())

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
                    output = model(prompt, TGAnswer)
                    pred_dst = int(json.loads(output)['destination_node'])
                except Exception as e:
                    logger.error(f'Generation failed: {e}')
                    pred_dst = None

                record = {
                    'id': n,
                    'src': int(srcs[i]),
                    'ts': int(times[i]),
                    'true_dst': int(dsts[i]),
                    'pred_dst': pred_dst,
                    'correct': (pred_dst == int(dsts[i])),
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                }
                f.write(json.dumps(record) + '\n')

                n += 1
                if n >= args.max_test_edges:
                    stop = True
                    break

            bg.extend(srcs.tolist(), dsts.tolist(), times.tolist())

    logger.info(f'Wrote {n} cached answers to {args.out}')


if __name__ == '__main__':
    main()
