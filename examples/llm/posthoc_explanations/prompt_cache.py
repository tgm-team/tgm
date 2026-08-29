"""Stage 2 of the post-hoc explanation pipeline: build explanation prompts.

Reads the answer cache produced by ``answer_cache.py`` and, for each edge,
builds a chat prompt that asks a strong LLM (e.g. GPT-4 class) to explain why
the TGTalker model produced its prediction and to categorize the reasoning. The
output JSONL is the input to ``generate_explanations.py``.

This stage has no heavy dependencies -- it is pure text manipulation.
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reasoning categories the judge model is asked to choose from. These mirror the
# common temporal-graph prediction heuristics discussed in the TGTalker paper.
CATEGORIES = [
    'repetition',  # repeats a past interaction partner of the source
    'recency',  # follows the most recent interaction(s)
    'popularity',  # picks a globally frequent destination
    'structural',  # multi-hop / shared-neighbor reasoning
    'uncertain',  # no clear signal
]

EXPLANATION_SYSTEM = (
    'You are an expert analyst of temporal graph link-prediction models. '
    'Given the context shown to a predictor, its prediction, and the ground '
    'truth, explain in 2-3 sentences why the model likely made its prediction, '
    'then classify the dominant reasoning into exactly one of these categories: '
    + ', '.join(CATEGORIES)
    + '. Respond as JSON: {"explanation": str, "category": str}.'
)


def build_explanation_messages(record: dict) -> list:
    """Turn one cached answer into chat messages for the judge model."""
    user = (
        f'CONTEXT SHOWN TO THE MODEL:\n{record["system_prompt"]}\n'
        f'{record["user_prompt"]}\n\n'
        f'MODEL PREDICTION (Destination Node): {record["pred_dst"]}\n'
        f'GROUND TRUTH (Destination Node): {record["true_dst"]}\n'
        f'PREDICTION CORRECT: {record["correct"]}\n'
    )
    return [
        {'role': 'system', 'content': EXPLANATION_SYSTEM},
        {'role': 'user', 'content': user},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build explanation prompts')
    parser.add_argument('--answers', type=str, default='answer_cache/answers.jsonl')
    parser.add_argument(
        '--out', type=str, default='answer_cache/explanation_prompts.jsonl'
    )
    parser.add_argument(
        '--only-incorrect',
        action='store_true',
        help='only build prompts for incorrect predictions',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = 0
    with open(args.answers) as fin, open(args.out, 'w') as fout:
        for line in fin:
            record = json.loads(line)
            if args.only_incorrect and record.get('correct'):
                continue
            out = {
                'custom_id': f'edge-{record["id"]}',
                'messages': build_explanation_messages(record),
            }
            fout.write(json.dumps(out) + '\n')
            n += 1
    logger.info(f'Wrote {n} explanation prompts to {args.out}')


if __name__ == '__main__':
    main()
