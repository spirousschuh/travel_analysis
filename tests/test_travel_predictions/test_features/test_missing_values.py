from pytest import approx

from travel_analysis.const import NINE, NINE_PRE
from travel_analysis.travel_predictions.features.missing_values import \
    MissingEstimator, MissingValues


def test_get_distribution(training_sample):
    # given
    estimator = MissingEstimator(outputCol=NINE_PRE)

    # when
    distribution = estimator.get_distribution(training_sample)

    # then
    assert distribution == approx([0.98, 0.01, 0, 0.01], abs=1e-2)


def test_get_distribution_from_missing_data(missing_sample):
    # given
    estimator = MissingEstimator(outputCol=NINE_PRE)

    # when
    distribution = estimator.get_distribution(missing_sample)

    # then
    assert distribution == approx([2 / 3, 1 / 6, 1 / 12, 1 / 12])


def test_get_missing_value_model(missing_sample):
    # given
    estimator = MissingEstimator(outputCol=NINE_PRE)

    # when
    trained_model = estimator.fit(missing_sample)

    # then
    assert trained_model.rides_per_day == approx([2 / 3, 1 / 6, 1 / 12, 1 / 12])


def test_get_missing_value_single_replacement():
    # given
    missing_model = MissingValues(distribution=[0.8, 0.1, 0.05, 0.05],
                                  outputCol=NINE)

    # when / then
    assert missing_model.single_replacement(80, 10., 5., 5.) == 80
    assert missing_model.single_replacement(None, 10., 5., 5.) == 80
    assert missing_model.single_replacement(80., None, 5., 5.) == 80
    assert missing_model.single_replacement(None, 20., 10., 10.) == 160
    assert missing_model.single_replacement(None, 10., 10., 5.) == 100
    assert missing_model.single_replacement(None, 10., 10., 10.) == 120


def test_get_missing_value_transformation(missing_sample):
    # given
    missing_model = MissingValues(distribution=[0.8, 0.1, 0.05, 0.05],
                                  outputCol=NINE)

    # when
    predictions_df = missing_model.transform(missing_sample)

    # then
    predictions = [row[NINE] for row in predictions_df.collect()]
    assert predictions == approx([80, 80])
