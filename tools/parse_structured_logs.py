import argparse
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        'Parse TGM debug logs and extract structured metrics. '
        'Performs mean reduction across duplicate log entries.'
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--log-file-path', type=str, required=True, help='Path to log file')
parser.add_argument(
    '--json-save-path', type=str, required=True, help='Path to save parsed logs'
)


def main(log_file_str: str, json_save_str: str) -> None:
    log_file_path = Path(log_file_str)
    print(f'Parsing log file: {log_file_path}')
    structured_metrics = parse_log_file(log_file_path)

    json_save_path = Path(json_save_str)
    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    json_save_path.write_text(json.dumps(structured_metrics, indent=2))
    print(f'Saved parsed metrics to {json_save_path}')


def parse_log_file(log_file_path: Path) -> dict:
    structured_logs = defaultdict(list)
    for line in log_file_path.read_text().splitlines():
        try:
            msg = extract_log_msg(line)
        except Exception as e:
            print(f'Failed to extract log msg from line: {line}')
            raise e

        if not (msg.startswith('{') and msg.endswith('}')):
            continue  # Skip non-JSON messages (could have some false positives)

        try:
            msg_dict = json.loads(msg)
        except json.JSONDecodeError as e:
            print(f'Failed to decode JSON for line: {line}')
            raise e

        msg_dict_keys = tuple(sorted(msg_dict.keys()))
        if msg_dict_keys != ('metric', 'value'):
            raise ValueError(
                'We only support parsing structured logs with keys: "metric" and "value", '
                f'found: {msg_dict_keys} in msg {msg} from line: {line}'
            )
        structured_logs[msg_dict['metric']].append(msg_dict['value'])

    # Perform mean reduction
    return {
        metric: sum(values) / len(values) for metric, values in structured_logs.items()
    }


def extract_log_msg(line: str) -> str:
    """Extract log message by skipping metadata.

    Logs have the form:
    [YYYY-MM-DD hh:mm:ss] LOG_NAME - LOG_LEVEL - [PROCESS THREAD FILE] LOG_MSG
    """
    return line[line.find(']', line.find(']') + 1) + 2 :]  # + 2 to skip '] '


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.log_file_path, args.json_save_path)
