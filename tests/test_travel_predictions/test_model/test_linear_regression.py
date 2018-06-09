from pyspark.ml import PipelineModel

from travel_analysis.const import NINE
from travel_analysis.travel_predictions.model.linear_regression import train


def test_linear_regression_training(training_sample):
    # given

    # when
    trained_model = train(training_sample)

    # then
    assert isinstance(trained_model, PipelineModel)


def test_predictions_of_the_model(sample_predictions):
    all_rows = [row.asDict() for row in sample_predictions.collect()]
    predictions = [row['prediction'] for row in all_rows]
    labels = [row[NINE] for row in all_rows]
    differences = [abs(prediction - label)
                   for (prediction, label) in zip(predictions, labels)]
    assert all([one_diff < 10 for one_diff in differences])
