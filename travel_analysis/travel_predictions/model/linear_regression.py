from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from travel_analysis.const import FEATURES, NINE, NINE_PRE, TWELF, TWELF_PRE, \
    PURE_FEATURES, TWELF_FEAT, FIFTEEN_FEAT, FIFTEEN, FIFTEEN_PRE, \
    NINETEEN_FEAT, NINETEEN, NINETEEN_PRE
from travel_analysis.travel_predictions.features.create_all import \
    get_features_pipeline


def train(pure_input):
    features = get_features_pipeline()

    lr_model9 = LinearRegression(featuresCol=PURE_FEATURES,
                                 labelCol=NINE,
                                 predictionCol=NINE_PRE)
    assembler_12 = VectorAssembler(inputCols=([PURE_FEATURES, NINE_PRE]),
                                   outputCol=TWELF_FEAT)
    lr_model12 = LinearRegression(featuresCol=TWELF_FEAT,
                                  labelCol=TWELF,
                                  predictionCol=TWELF_PRE)
    assembler_15 = VectorAssembler(
        inputCols=([PURE_FEATURES, NINE_PRE, TWELF_PRE]),
        outputCol=FIFTEEN_FEAT)
    lr_model15 = LinearRegression(featuresCol=FIFTEEN_FEAT,
                                  labelCol=FIFTEEN,
                                  predictionCol=FIFTEEN_PRE)
    assembler_19 = VectorAssembler(
        inputCols=([PURE_FEATURES, NINE_PRE, TWELF_PRE, FIFTEEN_PRE]),
        outputCol=NINETEEN_FEAT)
    lr_model19 = LinearRegression(featuresCol=NINETEEN_FEAT,
                                  labelCol=NINETEEN,
                                  predictionCol=NINETEEN_PRE)
    return Pipeline(stages=[features, lr_model9,
                            assembler_12, lr_model12,
                            assembler_15, lr_model15,
                            assembler_19, lr_model19,
                            ]).fit(pure_input)
