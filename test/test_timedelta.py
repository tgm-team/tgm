from opendg.timedelta import TimeDeltaTG


def test_timedelta():
    td1 = TimeDeltaTG('Y', 1)
    td2 = TimeDeltaTG('M', 1)

    assert td1.unit == 'Y'
    assert td1.value == 1

    rate1 = td1.convert('M')
    assert rate1 == (365 / 30)
    rate2 = td1.convert(td2)
    assert rate2 == rate1

    td3 = TimeDeltaTG('M', 3)
    rate = td1.convert(td3)
    assert rate == ((365 / 30) / 3)
