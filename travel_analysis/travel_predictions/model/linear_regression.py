from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

from travel_analysis.const import NINE, FEATURES, NINE_PREDICTION
from travel_analysis.travel_predictions.features.create_all import \
    get_features_pipeline


def train(pure_input):
    features = get_features_pipeline()
    lr_model = LinearRegression(featuresCol=FEATURES,
                                labelCol=NINE,
                                predictionCol=NINE_PREDICTION)
    return Pipeline(stages=[features, lr_model]).fit(pure_input)
