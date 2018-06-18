import datetime

from pytest import approx

from travel_analysis.const import HOLIDAY, BEFORE_HOLIDAY
from travel_analysis.travel_predictions.features.holiday import Holiday, \
    BeforeHoliday


def test_is_holiday():
    # given
    feature_creator = Holiday()
    test_dates = [datetime.datetime(2018, 12, 25),
                  datetime.datetime(2018, 12, 30),
                  datetime.datetime(2018, 5, 1),
                  ]

    # when
    holiday_feature = [feature_creator._is_holiday(day) for day in test_dates]

    # then
    assert holiday_feature == approx([1, 0, 1])


def test_holidays_for_dfs(training_sample):
    # given
    feature_creator = Holiday()

    # when
    df_with_holiday = feature_creator.transform(training_sample)

    # then
    holiday_feature = [row[HOLIDAY] for row in df_with_holiday.collect()]
    assert holiday_feature == approx([1, 0, 0, 0, 0, 0, 0, 1, 0, 0])


def test_before_holidays_for_dfs(training_sample):
    # given
    feature_creator = BeforeHoliday()

    # when
    df_with_holiday = feature_creator.transform(training_sample)

    # then
    holiday_feature = [row[BEFORE_HOLIDAY] for row in df_with_holiday.collect()]
    assert holiday_feature == approx([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
