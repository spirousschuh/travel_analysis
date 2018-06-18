from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, \
    StringIndexer, Bucketizer, PolynomialExpansion

from travel_analysis.const import WEEKDAY, DAY_VECTOR, NINE, TWELF, \
    FIFTEEN, NINETEEN, HOLIDAY, BEFORE_HOLIDAY, DIRECTION, DIRECTION_FLAG, \
    DIRECTION_IDX, RIDES_PER_DAY, HOUR, PURE_FEATURES, CAPACITY, DEPARTURE, \
    LABEL_COLS, ROLLING_AVGS, FRAC_9, FRAC_12, FRAC_15, FRAC_19, \
    FULL_TRAINING_DATA, FULL_TRAIN_DATA_FEATURES
from travel_analysis.travel_predictions.features.Hour import Hour
from travel_analysis.travel_predictions.features.fraction import Fraction
from travel_analysis.travel_predictions.features.holiday import Holiday, \
    BeforeHoliday
from travel_analysis.travel_predictions.features.missing_values import \
    MissingEstimator
from travel_analysis.travel_predictions.features.rides_per_day import \
    RidesPerDayEstimator
from travel_analysis.travel_predictions.features.rolling_average import \
    RollingAverageEstimator
from travel_analysis.travel_predictions.features.weekday import Weekday
from travel_analysis.utils.io_spark import get_dataframe

HOUR_BUCKET = 'hour_bucket'


def get_features_pipeline():
    feature_steps = [MissingEstimator(outputCol=NINE),
                     MissingEstimator(outputCol=TWELF),
                     MissingEstimator(outputCol=FIFTEEN),
                     MissingEstimator(outputCol=NINETEEN),
                     Holiday(),
                     # BeforeHoliday(),
                     # Hour(),
                     # Bucketizer(splits=[0, 8, 12, 17, 22, 24], inputCol=HOUR,
                     #            outputCol=HOUR_BUCKET),
                     Fraction(inputCols=[NINE, CAPACITY], outputCol=FRAC_9),
                     Fraction(inputCols=[TWELF, CAPACITY], outputCol=FRAC_12),
                     Fraction(inputCols=[FIFTEEN, CAPACITY], outputCol=FRAC_15),
                     Fraction(inputCols=[NINETEEN, CAPACITY], outputCol=FRAC_19),
                     RollingAverageEstimator(date_column=DEPARTURE,
                                             first_day_of_window=14,
                                             time_span_avg=7,
                                             inputCols=[FRAC_9, FRAC_12, FRAC_15, FRAC_19],
                                             outputCols=ROLLING_AVGS),
                     Weekday(),
                     # StringIndexer(inputCol=DIRECTION, outputCol=DIRECTION_IDX),
                     # OneHotEncoderEstimator(inputCols=[WEEKDAY, DIRECTION_IDX],
                     #                        outputCols=[DAY_VECTOR,
                     #                                    DIRECTION_FLAG]),
                     RidesPerDayEstimator(),
                     # VectorAssembler(
                     #     inputCols=([DAY_VECTOR, HOLIDAY, DIRECTION_FLAG,
                     #                 RIDES_PER_DAY,
                     #                 # HOUR_BUCKET, BEFORE_HOLIDAY,
                     #                 CAPACITY, *ROLLING_AVGS]
                     #     ),
                     #     outputCol='aggregated_features'),
                     # PolynomialExpansion(degree=2,
                     #                     inputCol='aggregated_features',
                     #                     outputCol=PURE_FEATURES)
                     ]

    return Pipeline(stages=feature_steps)


if __name__ == '__main__':
    trainings_data = get_dataframe(FULL_TRAINING_DATA)
    pipeline = get_features_pipeline()
    trained_pipeline = pipeline.fit(trainings_data)
    trained_pipeline.transform(trainings_data).write.csv(FULL_TRAIN_DATA_FEATURES, header=True)