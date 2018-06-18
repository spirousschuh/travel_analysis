import logging

from pyspark.ml.evaluation import RegressionEvaluator

from travel_analysis.const import NINETEEN_PRE, \
    FIFTEEN_PRE, TWELF_PRE, NINE_PRE, FULL_TRAINING_DATA, CAPACITY, \
    LABEL_COLS
from travel_analysis.travel_predictions.model.linear_regression import train
from travel_analysis.utils.io_spark import get_dataframe

logger = logging.getLogger(__name__)


def get_r2_metrics(input_df, label_col, prediction_col):
    evaluator = RegressionEvaluator(labelCol=label_col,
                                    predictionCol=prediction_col,
                                    metricName='r2')
    return evaluator.evaluate(input_df)


def evaluate_model_on_all_price_levels(input_features):
    filtered_input_features = filter_weird_observations(input_features)
    prediction_cols = [NINE_PRE,TWELF_PRE, FIFTEEN_PRE, NINETEEN_PRE]


    models = train(filtered_input_features)

    predictions = models.transform(input_features)

    r2_metrics = [get_r2_metrics(predictions, label_col, pred_col)
                  for (label_col, pred_col)
                  in zip(LABEL_COLS, prediction_cols)]
    logger.info('r2 metrics for 19, 15, 12, 9 € '
                'tickets is {r2s}'.format(r2s=r2_metrics))
    print(r2_metrics)
    print(sum(r2_metrics))
    return r2_metrics


def filter_weird_observations(input_features):
    columns = [getattr(input_features, col) for col in LABEL_COLS]
    nine, twelf, fifteen, nineteen = columns
    capacity = getattr(input_features, CAPACITY)
    filtered_input_features = input_features.filter(
        (nine + twelf + fifteen + nineteen) <= capacity)
    return filtered_input_features


if __name__ == '__main__':
    input_features = get_dataframe(FULL_TRAINING_DATA)

    evaluate_model_on_all_price_levels(input_features)
