import logging

from pyspark.ml.evaluation import RegressionEvaluator

from travel_analysis.const import NINE, TWELF, FIFTEEN, NINETEEN, \
    NINETEEN_PRE, \
    FIFTEEN_PRE, TWELF_PRE, NINE_PRE, FULL_TRAINING_DATA
from travel_analysis.travel_predictions.model.linear_regression import train
from travel_analysis.utils.io_spark import get_dataframe

logger = logging.getLogger(__name__)


def get_r2_metrics(input_df, label_col, prediction_col):
    evaluator = RegressionEvaluator(labelCol=label_col,
                                    predictionCol=prediction_col,
                                    metricName='r2')
    return evaluator.evaluate(input_df)


def evaluate_model_on_all_price_levels(input_features):
    label_cols = [NINETEEN, FIFTEEN, TWELF, NINE]
    prediction_cols = [NINETEEN_PRE, FIFTEEN_PRE, TWELF_PRE, NINE_PRE]

    models = [train(input_features, label, prediction)
              for (label, prediction) in zip(label_cols, prediction_cols)]

    predictions = [model.transform(input_features) for model in models]

    r2_metrics = [get_r2_metrics(pred, label_col, pred_col)
                  for (pred, label_col, pred_col)
                  in zip(predictions, label_cols, prediction_cols)]
    logger.info('r2 metrics for 19, 15, 12, 9 € '
                'tickets is {r2s}'.format(r2s=r2_metrics))
    print(r2_metrics)
    return r2_metrics


if __name__ == '__main__':
    input_features = get_dataframe(FULL_TRAINING_DATA)

    evaluate_model_on_all_price_levels(input_features)
