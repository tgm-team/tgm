#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Merge all JSON files in a directory into a single JSON file'
)
parser.add_argument(
    'log_dir', type=str, help='Directory containing JSON files to merge'
)
parser.add_argument(
    '--out_file_path',
    type=str,
    default='metrics.json',
    help='Output merged JSON filename (default: metrics.json)',
)


def main(log_dir_str: str, out_file_str: str) -> None:
    log_dir = Path(log_dir_str)
    if not log_dir.is_dir():
        raise NotADirectoryError(f'{log_dir} is not a directory')

    out_file = log_dir / out_file_str
    merged = {}
    for f in log_dir.glob('*.json'):
        if f.name != out_file.name:
            merged[f.stem] = json.loads(f.read_text())
    out_file.write_text(json.dumps(merged, indent=2))
    print(f'Merged JSON saved to {out_file}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.log_dir, args.out_file_path)
