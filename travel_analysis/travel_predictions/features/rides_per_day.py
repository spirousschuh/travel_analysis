from operator import attrgetter

from pyspark.ml import Estimator, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, \
    TypeConverters
from pyspark.sql.functions import udf
from pyspark.sql.types import DateType, IntegerType

from travel_analysis.const import RIDES_PER_DAY, DEPARTURE


class RidesPerDayEstimator(Estimator, HasInputCol, HasOutputCol):
    def __init__(self, outputCol=RIDES_PER_DAY,
                 inputCol=DEPARTURE,
                 ):
        super().__init__()
        self._setDefault(outputCol=outputCol,
                         inputCol=inputCol)
        self._set(outputCol=outputCol,
                  inputCol=inputCol)

    def get_rides_per_day(self, dataset):
        date_col = self.getInputCol()
        dates = dataset.select(attrgetter(date_col)(dataset).cast(DateType()))
        aggregation_row = dates.groupby(date_col).count()
        return {row[date_col]: row['count']
                for row in aggregation_row.collect()}

    def _fit(self, dataset):
        look_up = self.get_rides_per_day(dataset)
        return RidesPerDay(look_up,
                           self.getOutputCol(),
                           self.getInputCol())


class RidesPerDay(Transformer, HasInputCol, HasOutputCol):
    def __init__(self,
                 rides_per_day,
                 outputCol=RIDES_PER_DAY,
                 inputCol=DEPARTURE,
                 ):
        super().__init__()
        self._setDefault(_rides_per_day=rides_per_day,
                         outputCol=outputCol,
                         inputCol=inputCol)
        self._set(_rides_per_day=rides_per_day,
                  outputCol=outputCol,
                  inputCol=inputCol)

    _rides_per_day = Param(Params._dummy(), '_rides_per_day',
                           "rides_per_day of seats over ticket prices ",
                           typeConverter=TypeConverters.identity)

    @property
    def rides_per_day(self):
        """
        Gets the value of the Spark parameter for the rides_per_day.
        """
        return self.getOrDefault(self._rides_per_day)

    def single_replacement(self, date_and_time):
        return self.rides_per_day[date_and_time.date()]

    def _transform(self, dataset):
        spark_replace = udf(self.single_replacement, IntegerType())
        return dataset.withColumn(self.getOutputCol(),
                                  spark_replace(self.getInputCol()))
