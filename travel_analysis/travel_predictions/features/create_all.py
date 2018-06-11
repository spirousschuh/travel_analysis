from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler

from travel_analysis.const import WEEKDAY, DAY_VECTOR, FEATURES, NINE, TWELF, \
    FIFTEEN, NINETEEN
from travel_analysis.travel_predictions.features.missing_values import \
    MissingEstimator
from travel_analysis.travel_predictions.features.weekday import Weekday


def get_features_pipeline():
    feature_steps = [MissingEstimator(outputCol=NINE),
                     MissingEstimator(outputCol=TWELF),
                     MissingEstimator(outputCol=FIFTEEN),
                     MissingEstimator(outputCol=NINETEEN),
                     Weekday(),
                     OneHotEncoderEstimator(inputCols=[WEEKDAY],
                                            outputCols=[DAY_VECTOR],
                                            dropLast=False),
                     VectorAssembler(inputCols=[DAY_VECTOR], outputCol=FEATURES)
                     ]

    return Pipeline(stages=feature_steps)
