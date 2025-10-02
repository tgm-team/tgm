import json
import tempfile

import tgm.util.logging as tgm_logging


@tgm_logging.log_latency
def dummy_latency():
    return 42


@tgm_logging.log_gpu
def dummy_gpu():
    return 42


def dummy_metrics():
    tgm_logging.log_metric('foo', 123, epoch=1)
    tgm_logging.log_metric('bar', 234)


def dummy_metrics_dict():
    tgm_logging.log_metrics_dict({'foo': 123, 'bar': 234}, epoch=1)


def test_log_latency_logging_disabled():
    with tempfile.NamedTemporaryFile() as tmp_file:
        dummy_latency()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Function dummy_latency executed in ' not in terminal


def test_log_gpu_logging_disabled():
    with tempfile.NamedTemporaryFile() as tmp_file:
        dummy_gpu()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Function dummy_gpu GPU memory in ' not in terminal


def test_log_metrics_logging_disabled():
    with tempfile.NamedTemporaryFile() as tmp_file:
        dummy_metrics()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Epoch=02 foo=123' not in terminal
        # TODO: assert 'bar=234' not in terminal


def test_log_metrics_dict_logging_disabled():
    with tempfile.NamedTemporaryFile() as tmp_file:
        dummy_metrics_dict()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Epoch=02 foo=123' not in terminal
        # TODO: assert 'bar=234' not in terminal


def test_log_latency_console_only():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        dummy_latency()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Function dummy_latency executed in ' is in terminal


def test_log_gpu_console_only():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        dummy_gpu()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Function dummy_gpu GPU memory in ' is in terminal


def test_log_metrics_console_only():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        dummy_metrics()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Epoch=02 foo=123' is in terminal
        # TODO: assert 'bar=234' is in terminal


def test_log_metrics_dict_console_only():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        dummy_metrics_dict()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]
        assert len(lines) == 0

        # TODO: assert 'Epoch=02 foo=123' is in terminal
        # TODO: assert 'bar=234' is in terminal


def test_log_latency_console_and_file():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        dummy_latency()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]

        # TODO: assert 'Function dummy_latency executed in ' is in terminal

        json_lines = [line for line in lines if line.endswith('}')]
        json_lines = [
            json.loads(line[line.find('{') : line.rfind('}') + 1])
            for line in json_lines
        ]

        assert len(json_lines) == 1
        assert set(['metric', 'value', 'function']) == set(json_lines[0].keys())
        assert json_lines[0]['metric'] == 'dummy_latency_latency'
        assert json_lines[0]['function'] == 'dummy_latency'


def test_log_gpu_console_and_file():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        dummy_gpu()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]

        # TODO: assert 'Function dummy_gpu GPU memory in ' is in terminal

        json_lines = [line for line in lines if line.endswith('}')]
        json_lines = [
            json.loads(line[line.find('{') : line.rfind('}') + 1])
            for line in json_lines
        ]

        assert len(json_lines) == 1
        assert set(
            [
                'metric',
                'cuda_available',
                'start_mb',
                'peak_mb',
                'end_mb',
                'diff_mb',
                'function',
            ]
        ) == set(json_lines[0].keys())
        assert json_lines[0]['metric'] == 'dummy_gpu_gpu_usage'
        assert json_lines[0]['function'] == 'dummy_gpu'


def test_log_metrics_console_and_file():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        dummy_metrics()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]

        # TODO: assert 'Epoch=02 foo=123' is in terminal
        # TODO: assert 'bar=234' is in terminal

        json_lines = [line for line in lines if line.endswith('}')]
        json_lines = [
            json.loads(line[line.find('{') : line.rfind('}') + 1])
            for line in json_lines
        ]

        assert len(json_lines) == 2
        assert json_lines[0] == {'metric': 'foo', 'value': 123, 'epoch': 1}
        assert json_lines[1] == {'metric': 'bar', 'value': 234}


def test_log_metrics_dict_console_and_file():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        dummy_metrics_dict()
        tmp_file.seek(0)
        lines = [line.decode() for line in tmp_file.read().splitlines()]

        # TODO: assert 'Epoch=02 foo=123' is in terminal
        # TODO: assert 'bar=234' is in terminal

        json_lines = [line for line in lines if line.endswith('}')]
        json_lines = [
            json.loads(line[line.find('{') : line.rfind('}') + 1])
            for line in json_lines
        ]

        assert len(json_lines) == 2
        assert json_lines[0] == {'metric': 'foo', 'value': 123, 'epoch': 1}
        assert json_lines[1] == {'metric': 'bar', 'value': 234, 'epoch': 1}
