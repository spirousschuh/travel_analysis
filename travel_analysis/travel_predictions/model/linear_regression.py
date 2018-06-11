from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

from travel_analysis.const import FEATURES
from travel_analysis.travel_predictions.features.create_all import \
    get_features_pipeline


def train(pure_input, label_col, prediction_col):
    features = get_features_pipeline()
    lr_model = LinearRegression(featuresCol=FEATURES,
                                labelCol=label_col,
                                predictionCol=prediction_col)
    return Pipeline(stages=[features, lr_model]).fit(pure_input)
