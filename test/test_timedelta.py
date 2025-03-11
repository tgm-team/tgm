import pytest

from opendg.timedelta import TimeDeltaDG, TimeDeltaUnit


@pytest.fixture(params=['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def time_granularity(request):
    return request.param


def test_init_default_value(time_granularity):
    td = TimeDeltaDG(time_granularity)
    assert td.unit == time_granularity
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{time_granularity}', value=1)"
    assert not td.is_ordered


def test_init_non_default_value(time_granularity):
    value = 5
    td = TimeDeltaDG(time_granularity, value)
    assert td.unit == time_granularity
    assert td.value == value
    assert str(td) == f"TimeDeltaDG(unit='{time_granularity}', value={value})"
    assert not td.is_ordered


def test_init_ordered():
    time_granularity = TimeDeltaUnit.ORDERED
    td = TimeDeltaDG(time_granularity)
    assert td.unit == time_granularity
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{time_granularity}')"
    assert td.is_ordered


def test_init_with_time_delta_unit(time_granularity):
    time_granularity = TimeDeltaUnit(time_granularity)
    td = TimeDeltaDG(time_granularity)
    assert td.unit == time_granularity
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{time_granularity}', value=1)"
    assert not td.is_ordered


def test_init_ordered_with_time_delta_unit():
    time_granularity = TimeDeltaUnit.ORDERED
    td = TimeDeltaDG(time_granularity)
    assert td.unit == time_granularity
    assert td.value == 1
    assert str(td) == f"TimeDeltaDG(unit='{time_granularity}')"
    assert td.is_ordered


@pytest.mark.parametrize('bad_unit', ['mock'])
def test_init_bad_unit(bad_unit):
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(bad_unit)


@pytest.mark.parametrize('bad_value', [-1, 0])
def test_init_bad_value(time_granularity, bad_value):
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(time_granularity, bad_value)


@pytest.mark.parametrize('bad_value', [-1, 0, 2])
def test_init_ordered_bad_value(bad_value):
    # Note: Only '1' accepted as the value for an ordered type
    time_granularity = TimeDeltaUnit.ORDERED
    with pytest.raises(ValueError):
        _ = TimeDeltaDG(time_granularity, bad_value)


def test_convert_between_same_units(time_granularity):
    td1 = TimeDeltaDG(time_granularity, 2)
    td2 = TimeDeltaDG(time_granularity, 3)
    assert td1.convert(td2) == 2 / 3
    assert td2.convert(td1) == 3 / 2

    td1 = TimeDeltaDG(time_granularity, 1)
    td2 = TimeDeltaDG(time_granularity, 1)
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
    td1 = TimeDeltaDG(TimeDeltaUnit.ORDERED)
    td2 = TimeDeltaDG('Y', 1)

    with pytest.raises(ValueError):
        _ = td2.convert(td1)


def test_bad_convert_to_ordered():
    td1 = TimeDeltaDG('Y', 1)
    td2 = TimeDeltaDG(TimeDeltaUnit.ORDERED)

    with pytest.raises(ValueError):
        _ = td1.convert(td2)


def test_time_delta_is_coarser_than():
    td1 = TimeDeltaDG('s', value=30)
    td2 = TimeDeltaDG('m')
    assert td2.is_coarser_than(td1)
    assert not td1.is_coarser_than('m')

    # Check that comparison is strict
    td1 = TimeDeltaDG('s')
    assert not td1.is_coarser_than(td1)
    assert not td1.is_coarser_than('s')

    # Check that the 'value' is taken into account (not just unit)
    td1 = TimeDeltaDG('s', value=60 * 5)
    td2 = TimeDeltaDG('m', value=3)
    assert td1.is_coarser_than(td2)
    assert not td2.is_coarser_than(td1)


def test_time_delta_is_coarser_than_due_to_value():
    # Testing that granularity is not just based on the unit.
    # In this case, 1 minute should be more granular than 80 seconds
    # even though TimeDeltaUnit.SECOND is more granular than TimeDeltaUnit.MINUTE
    td1 = TimeDeltaDG('s', value=80)
    td2 = TimeDeltaDG('m')
    assert not td2.is_coarser_than(td1)
    assert td1.is_coarser_than(td2)


def test_time_delta_is_coarser_try_compare_ordered():
    td1 = TimeDeltaDG('r')
    td2 = TimeDeltaDG('s')

    with pytest.raises(ValueError):
        td1.is_coarser_than(td2)

    with pytest.raises(ValueError):
        td2.is_coarser_than(td1)

    with pytest.raises(ValueError):
        td1.is_coarser_than('r')


def test_time_unit_ordering():
    # This ensures consistent ordering behaviour between units.
    # The convention we use is that we order from more granular (smaller unit) to less granular (larger unit).
    units = TimeDeltaUnit._member_names_
    assert units == [
        'ORDERED',
        'NANOSECOND',
        'MICROSECOND',
        'MILLISECOND',
        'SECOND',
        'MINUTE',
        'HOUR',
        'DAY',
        'WEEK',
        'MONTH',
        'YEAR',
    ]


def test_time_unit_is_coarser_than():
    # Start i at index 1 since ORDERED enum value is not comparable (see test cases below)
    for i in range(1, len(TimeDeltaUnit._member_names_)):
        for j in range(i, len(TimeDeltaUnit._member_names_)):
            unit_i = TimeDeltaUnit.from_string(TimeDeltaUnit._member_names_[i])
            unit_j = TimeDeltaUnit.from_string(TimeDeltaUnit._member_names_[j])

            if i == j:
                assert not unit_i.is_coarser_than(unit_j)
                assert not unit_j.is_coarser_than(unit_i)
            else:
                assert not unit_i.is_coarser_than(unit_j)
                assert unit_j.is_coarser_than(unit_i)


def test_time_unit_is_coarser_than_with_string():
    # Start i at index 1 since ORDERED enum value is not comparable (see test cases below)
    for i in range(1, len(TimeDeltaUnit._member_names_)):
        for j in range(i, len(TimeDeltaUnit._member_names_)):
            unit_i = TimeDeltaUnit.from_string(TimeDeltaUnit._member_names_[i])
            unit_j = TimeDeltaUnit.from_string(TimeDeltaUnit._member_names_[j])

            if i == j:
                assert not unit_i.is_coarser_than(TimeDeltaUnit._member_names_[j])
                assert not unit_j.is_coarser_than(TimeDeltaUnit._member_names_[i])
            else:
                assert not unit_i.is_coarser_than(TimeDeltaUnit._member_names_[j])
                assert unit_j.is_coarser_than(TimeDeltaUnit._member_names_[i])


def test_time_unit_is_coarser_than_with_ordered(time_granularity):
    unit = TimeDeltaUnit.from_string(time_granularity)
    with pytest.raises(ValueError):
        unit.is_coarser_than(TimeDeltaUnit.ORDERED)

    unit = TimeDeltaUnit.from_string(time_granularity)
    with pytest.raises(ValueError):
        TimeDeltaUnit.ORDERED.is_coarser_than(unit)


def test_time_unit_is_coarser_than_with_ordered_string(time_granularity):
    unit = TimeDeltaUnit.from_string(time_granularity)
    with pytest.raises(ValueError):
        unit.is_coarser_than('r')

    with pytest.raises(ValueError):
        TimeDeltaUnit.ORDERED.is_coarser_than(time_granularity)
