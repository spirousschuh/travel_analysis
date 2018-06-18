import datetime
from pytest import approx

from travel_analysis.const import DEPARTURE, LABEL_COLS, ROLLING_AVGS, AVG_9, \
    AVG_15
from travel_analysis.travel_predictions.features.rolling_average import \
    RollingAverage, RollingAverageEstimator


def test_rolling_average(training_sample):
    # given
    feature_creator = RollingAverageEstimator(date_column=DEPARTURE,
                                              first_day_of_window=3,
                                              time_span_avg=3,
                                              inputCols=LABEL_COLS,
                                              outputCols=ROLLING_AVGS)

    # when
    avgs_per_date = feature_creator.get_rolling_avgs_per_date(training_sample)

    # then
    expected_avgs = approx({datetime.date(2015, 1, 1): [None, None, None, None],
                            datetime.date(2015, 1, 2): [21.0, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 3): [16.5, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 4): [22.0, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 5): [24.5, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 6): [39 + 1 / 3, 0.0, 0.0,
                                                        0.0],
                            datetime.date(2015, 1, 7): [54.0, 1 / 3, 1 / 3,
                                                        2 / 3]
                            })
    assert expected_avgs == avgs_per_date


def test_rolling_average_training(training_sample):
    # given
    feature_creator = RollingAverageEstimator(date_column=DEPARTURE,
                                              first_day_of_window=3,
                                              time_span_avg=3,
                                              inputCols=LABEL_COLS,
                                              outputCols=ROLLING_AVGS)

    # when
    trained_model = feature_creator.fit(training_sample)

    # then
    expected_avgs = approx({datetime.date(2015, 1, 1): [None, None, None, None],
                            datetime.date(2015, 1, 2): [21.0, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 3): [16.5, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 4): [22.0, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 5): [24.5, 0.0, 0.0, 0.0],
                            datetime.date(2015, 1, 6): [39 + 1 / 3, 0.0, 0.0,
                                                        0.0],
                            datetime.date(2015, 1, 7): [54.0, 1 / 3, 1 / 3,
                                                        2 / 3]
                            })
    assert expected_avgs == trained_model.avgs_per_date


def test_rolling_average_transformation(training_sample):
    # given
    feature_creator = RollingAverageEstimator(date_column=DEPARTURE,
                                              first_day_of_window=3,
                                              time_span_avg=3,
                                              inputCols=LABEL_COLS,
                                              outputCols=ROLLING_AVGS)

    trained_model = feature_creator.fit(training_sample)

    # when
    df_with_avgs = trained_model.transform(training_sample)

    # then
    rows = [row for row in df_with_avgs.collect()]
    expected_9_avgs = approx(
        [0, 21.0, 16.5, 22.0, 22.0, 24.5, 24.5, 39 + 1 / 3, 54., 54.])
    avgs_9 = [row[AVG_9] for row in rows]
    assert expected_9_avgs == avgs_9
    expected_15_avgs = approx([0, 0, 0, 0, 0, 0, 0, 0, 1 / 3, 1 / 3])
    avgs_15 = [row[AVG_15] for row in rows]
    assert avgs_15 == expected_15_avgs
