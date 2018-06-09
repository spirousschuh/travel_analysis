from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import dayofweek, to_date

from travel_analysis.const import DEPARTURE, WEEKDAY


class Weekday(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, inputCol=DEPARTURE, outputCol=WEEKDAY):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(),
                                  dayofweek(to_date(self.getInputCol())))
