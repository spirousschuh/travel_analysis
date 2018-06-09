from pyspark.ml.evaluation import RegressionEvaluator


def get_r2_metrics(input_df, label_col, prediction_col):
    evaluator = RegressionEvaluator(labelCol=label_col,
                                    predictionCol=prediction_col,
                                    metricName='r2')
    return evaluator.evaluate(input_df)