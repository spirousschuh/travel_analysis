import datetime

from pytest import approx

from travel_analysis.const import RIDES_PER_DAY
from travel_analysis.travel_predictions.features.rides_per_day import \
    RidesPerDayEstimator


def test_get_rides_per_day(training_sample):
    # given
    feature_creator = RidesPerDayEstimator()

    # when
    counts = feature_creator.get_rides_per_day(training_sample)

    # then
    expected_counts = {datetime.date(2015, 1, 1): 1,
                       datetime.date(2015, 1, 2): 1,
                       datetime.date(2015, 1, 3): 1,
                       datetime.date(2015, 1, 4): 2,
                       datetime.date(2015, 1, 5): 2,
                       datetime.date(2015, 1, 6): 1,
                       datetime.date(2015, 1, 7): 2,
                       }
    assert counts == expected_counts


def test_rides_per_day_mapping(training_sample):
    # given
    feature_estimator = RidesPerDayEstimator()

    # when
    feature_creator = feature_estimator.fit(training_sample)
    df_with_counts = feature_creator.transform(training_sample)

    # then

    counts = [row[RIDES_PER_DAY] for row in df_with_counts.collect()]
    assert counts == approx([1, 1, 1, 2, 2, 2, 2, 1, 2, 2])
