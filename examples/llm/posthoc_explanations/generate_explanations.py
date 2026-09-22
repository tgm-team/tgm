"""Stage 3 of the post-hoc explanation pipeline: generate explanations.

Sends the prompts built by ``prompt_cache.py`` to an OpenAI chat model
(GPT-4 class) and writes the explanations + reasoning categories to JSONL,
along with a per-category summary count.

Requires the ``openai`` package and the ``OPENAI_API_KEY`` environment variable.
This stage is intentionally not exercised in CI (it makes paid external API
calls).
"""

import argparse
import collections
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate explanations via OpenAI')
    parser.add_argument(
        '--prompts', type=str, default='answer_cache/explanation_prompts.jsonl'
    )
    parser.add_argument('--out', type=str, default='output/explanations.jsonl')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument(
        '--limit', type=int, default=None, help='max prompts to process'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        logger.error(
            'OPENAI_API_KEY is not set. Export it before running this stage:\n'
            '  export OPENAI_API_KEY=sk-...\n'
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error('The `openai` package is required: pip install openai')
        sys.exit(1)

    client = OpenAI()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    category_counts: collections.Counter = collections.Counter()
    n = 0
    with open(args.prompts) as fin, open(args.out, 'w') as fout:
        for line in fin:
            if args.limit is not None and n >= args.limit:
                break
            item = json.loads(line)
            resp = client.chat.completions.create(
                model=args.model,
                messages=item['messages'],
                response_format={'type': 'json_object'},
            )
            content = resp.choices[0].message.content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {'explanation': content, 'category': 'uncertain'}
            category_counts[parsed.get('category', 'uncertain')] += 1
            fout.write(json.dumps({'custom_id': item['custom_id'], **parsed}) + '\n')
            n += 1

    logger.info(f'Wrote {n} explanations to {args.out}')
    logger.info(f'Category distribution: {dict(category_counts)}')


if __name__ == '__main__':
    main()
