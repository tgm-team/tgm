from tgm.util._tgb import suppress_output


def test_suppress_output():
    def mock_func_with_arg(x):
        print('raw file found')
        return True

    def mock_func_without_arg():
        print('raw file found')
        return True

    assert suppress_output(mock_func_with_arg, x=2)
    assert suppress_output(mock_func_without_arg)
