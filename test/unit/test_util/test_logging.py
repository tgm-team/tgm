import io
import json
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

import tgm.util.logging as tgm_logging


# TODO: Setup caplog fixture properly instead of monkey patching stdout
def capture_log_output(func, *args, **kwargs):
    stream = io.StringIO()
    with redirect_stdout(stream):
        func(*args, **kwargs)
        print(stream.getvalue())


@tgm_logging.log_latency
def dummy_latency():
    return 42


@tgm_logging.log_latency()
def dummy_latency_with_parens():
    return 42


@tgm_logging.log_gpu
def dummy_gpu():
    return 42


def dummy_metrics():
    tgm_logging.log_metric('foo', 123, epoch=1)
    tgm_logging.log_metric('bar', 234, extra={'meta_data': 'foo'})


def dummy_metrics_dict():
    tgm_logging.log_metrics_dict({'foo': 123, 'bar': 234}, epoch=1)


def test_log_latency_logging_disabled(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        capture_log_output(dummy_latency)

        console_logs = capsys.readouterr().err
        assert console_logs == ''

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_gpu_logging_disabled(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        capture_log_output(dummy_gpu)

        console_logs = capsys.readouterr().err
        assert console_logs == ''

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_metrics_logging_disabled(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        capture_log_output(dummy_metrics)

        console_logs = capsys.readouterr().err
        assert console_logs == ''

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_metrics_dict_logging_disabled(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        capture_log_output(dummy_metrics_dict)

        console_logs = capsys.readouterr().err
        assert console_logs == ''

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_latency_console_only(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        capture_log_output(dummy_latency)

        console_logs = capsys.readouterr().err
        assert 'Function dummy_latency executed in ' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_latency_with_parens_console_only(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        capture_log_output(dummy_latency_with_parens)

        console_logs = capsys.readouterr().err
        assert 'Function dummy_latency_with_parens executed in ' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_gpu_console_only(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        capture_log_output(dummy_gpu)

        console_logs = capsys.readouterr().err
        assert 'Function dummy_gpu GPU memory' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_gpu_withou_cuda_available(capsys):
    with patch('torch.cuda.is_available', return_value=False):
        with tempfile.NamedTemporaryFile() as tmp_file:
            tgm_logging.enable_logging()
            capture_log_output(dummy_gpu)

            console_logs = capsys.readouterr().err
            assert 'Function dummy_gpu GPU memory' in console_logs

            json_lines = _parse_json(tmp_file)
            assert len(json_lines) == 0


def test_log_metrics_console_only(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        capture_log_output(dummy_metrics)

        console_logs = capsys.readouterr().err
        assert 'Epoch=01 foo=123' in console_logs
        assert 'bar=234' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_metrics_dict_console_only(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging()
        capture_log_output(dummy_metrics_dict)

        console_logs = capsys.readouterr().err
        assert 'Epoch=01 foo=123' in console_logs
        assert 'Epoch=01 bar=234' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 0


def test_log_latency_console_and_file(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        capture_log_output(dummy_latency)

        console_logs = capsys.readouterr().err
        assert 'Function dummy_latency executed in ' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 1
        assert set(['metric', 'value']) == set(json_lines[0].keys())
        assert json_lines[0]['metric'] == 'dummy_latency latency'


def test_log_gpu_console_and_file(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        capture_log_output(dummy_gpu)

        console_logs = capsys.readouterr().err
        assert 'Function dummy_gpu GPU memory' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 2
        assert set(['metric', 'value']) == set(json_lines[0].keys())
        assert set(['metric', 'value']) == set(json_lines[1].keys())
        assert json_lines[0]['metric'] == 'dummy_gpu peak_gpu_mb'
        assert json_lines[0]['value'] == 0.0

        assert json_lines[1]['metric'] == 'dummy_gpu alloc_gpu_mb'
        assert json_lines[1]['value'] == 0.0


def test_log_metrics_console_and_file(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        capture_log_output(dummy_metrics)

        console_logs = capsys.readouterr().err
        assert 'Epoch=01 foo=123' in console_logs
        assert 'bar=234' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 2
        assert json_lines[0] == {'metric': 'foo epoch 1', 'value': 123}
        assert json_lines[1] == {'metric': 'bar', 'value': 234, 'meta_data': 'foo'}


def test_log_metrics_dict_console_and_file(capsys):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tgm_logging.enable_logging(log_file_path=tmp_file.name)
        capture_log_output(dummy_metrics_dict)

        console_logs = capsys.readouterr().err
        assert 'Epoch=01 foo=123' in console_logs
        assert 'Epoch=01 bar=234' in console_logs

        json_lines = _parse_json(tmp_file)
        assert len(json_lines) == 2
        assert json_lines[0] == {'metric': 'foo epoch 1', 'value': 123}
        assert json_lines[1] == {'metric': 'bar epoch 1', 'value': 234}


def test_pretty_number_format():
    assert tgm_logging.pretty_number_format(None) == 'None'
    assert tgm_logging.pretty_number_format(123) == '123'
    assert tgm_logging.pretty_number_format(123.45) == '123.45'
    assert tgm_logging.pretty_number_format(12345) == '12,345'
    assert tgm_logging.pretty_number_format(9876543210) == '9.88B'
    assert tgm_logging.pretty_number_format(float('inf')) == 'inf'
    assert tgm_logging.pretty_number_format(float('-inf')) == '-inf'
    assert tgm_logging.pretty_number_format(float('nan')) == 'nan'


def _parse_json(tmp_file):
    lines = [line.decode() for line in tmp_file.read().splitlines()]
    json_lines = [line for line in lines if line.endswith('}')]
    return [
        json.loads(line[line.find('{') : line.rfind('}') + 1]) for line in json_lines
    ]
