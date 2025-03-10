import pytest

from opendg.timedelta import TimeDeltaDG, TimeDeltaUnit


@pytest.fixture(params=['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def temporal_unit(request):
    return request.param


def test_init_default_value(temporal_unit):
    td = TimeDeltaDG(temporal_unit)
    assert td.unit == temporal_unit
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{temporal_unit}', value=1)"
    assert not td.is_ordered


def test_init_non_default_value(temporal_unit):
    value = 5
    td = TimeDeltaDG(temporal_unit, value)
    assert td.unit == temporal_unit
    assert td.value == value
    assert str(td) == f"TimeDeltaDG(unit='{temporal_unit}', value={value})"
    assert not td.is_ordered


def test_init_ordered():
    temporal_unit = 'r'
    td = TimeDeltaDG(temporal_unit)
    assert td.unit == temporal_unit
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{temporal_unit}')"
    assert td.is_ordered


def test_init_with_time_delta_unit(temporal_unit):
    temporal_unit = TimeDeltaUnit(temporal_unit)
    td = TimeDeltaDG(temporal_unit)
    assert td.unit == temporal_unit
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{temporal_unit}', value=1)"
    assert not td.is_ordered


def test_init_ordered_with_time_delta_unit():
    temporal_unit = TimeDeltaUnit.ORDERED
    td = TimeDeltaDG(temporal_unit)
    assert td.unit == temporal_unit
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{temporal_unit}')"
    assert td.is_ordered


@pytest.mark.parametrize('bad_unit', ['mock'])
def test_init_bad_unit(bad_unit):
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(bad_unit)


@pytest.mark.parametrize('bad_value', [-1, 0])
def test_init_bad_value(temporal_unit, bad_value):
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(temporal_unit, bad_value)


@pytest.mark.parametrize('bad_value', [-1, 0, 2])
def test_init_ordered_bad_value(bad_value):
    # Note: Only '1' accepted as the value for an ordered type
    temporal_unit = 'r'
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(temporal_unit, bad_value)


def test_convert_between_same_units(temporal_unit):
    td1 = TimeDeltaDG(temporal_unit, 2)
    td2 = TimeDeltaDG(temporal_unit, 3)
    assert td1.convert(td2) == 2 / 3
    assert td2.convert(td1) == 3 / 2

    td1 = TimeDeltaDG(temporal_unit, 1)
    td2 = TimeDeltaDG(temporal_unit, 1)
    assert td1.convert(td2) == 1
    assert td2.convert(td1) == 1


def test_convert_into_different_units():
    value = 5

    td = TimeDeltaDG('us', value)
    assert td.convert('ns') == value * 1000

    td = TimeDeltaDG('ms', value)
    assert td.convert('ns') == value * 1000 * 1000

    td = TimeDeltaDG('s', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000

    td = TimeDeltaDG('m', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60

    td = TimeDeltaDG('h', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60 * 60

    td = TimeDeltaDG('D', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60 * 60 * 24

    td = TimeDeltaDG('W', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60 * 60 * 24 * 7

    td = TimeDeltaDG('M', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60 * 60 * 24 * 30

    td = TimeDeltaDG('Y', value)
    assert td.convert('ns') == value * 1000 * 1000 * 1000 * 60 * 60 * 24 * 365


def test_convert_between_different_units():
    value_a = 5
    value_b = 3

    td_a = TimeDeltaDG('ns', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == (value_a / value_b) / (1000 * 1000 * 1000)
    assert td_b.convert(td_a) == (1000 * 1000 * 1000) * (value_b / value_a)

    td_a = TimeDeltaDG('us', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == (value_a / value_b) / (1000 * 1000)
    assert td_b.convert(td_a) == (1000 * 1000) * (value_b / value_a)

    td_a = TimeDeltaDG('ms', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == (value_a / value_b) / (1000)
    assert td_b.convert(td_a) == (1000) * (value_b / value_a)

    td_a = TimeDeltaDG('s', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 1 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / 1

    td_a = TimeDeltaDG('m', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / 60

    td_a = TimeDeltaDG('h', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 60 * 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / (60 * 60)

    td_a = TimeDeltaDG('D', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 24 * 60 * 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / (24 * 60 * 60)

    td_a = TimeDeltaDG('W', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 7 * 24 * 60 * 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / (7 * 24 * 60 * 60)

    td_a = TimeDeltaDG('M', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 30 * 24 * 60 * 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / (30 * 24 * 60 * 60)

    td_a = TimeDeltaDG('Y', value_a)
    td_b = TimeDeltaDG('s', value_b)
    assert td_a.convert(td_b) == 365 * 24 * 60 * 60 * (value_a / value_b)
    assert td_b.convert(td_a) == (value_b / value_a) / (365 * 24 * 60 * 60)


def test_bad_convert_from_ordered():
    td1 = TimeDeltaDG('r')
    td2 = TimeDeltaDG('Y', 1)

    with pytest.raises(ValueError):
        _ = td2.convert(td1)


def test_bad_convert_to_ordered():
    td1 = TimeDeltaDG('Y', 1)
    td2 = TimeDeltaDG('r')

    with pytest.raises(ValueError):
        _ = td1.convert(td2)
