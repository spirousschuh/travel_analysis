from datetime import timedelta

import holidays
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCol, HasInputCol
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from travel_analysis.const import DEPARTURE, BEFORE_HOLIDAY
from travel_analysis.const import HOLIDAY


class Holiday(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, inputCol=DEPARTURE, outputCol=HOLIDAY):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.holiday = holidays.Germany(prov='BY')

    def _is_holiday(self, date):
        return float(date in self.holiday)

    def _transform(self, dataset):
        spark_holiday = udf(self._is_holiday, FloatType())
        return dataset.withColumn(self.getOutputCol(),
                                  spark_holiday(self.getInputCol()))


class BeforeHoliday(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, inputCol=DEPARTURE, outputCol=BEFORE_HOLIDAY):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.holiday = holidays.Germany(prov='BY')

    def _is_before_holiday(self, date):
        return float((timedelta(days=1) + date) in self.holiday)

    def _transform(self, dataset):
        spark_holiday = udf(self._is_before_holiday, FloatType())
        return dataset.withColumn(self.getOutputCol(),
                                  spark_holiday(self.getInputCol()))
