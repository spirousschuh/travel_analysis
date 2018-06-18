from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import hour

from travel_analysis.const import DEPARTURE, HOUR


class Hour(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, inputCol=DEPARTURE, outputCol=HOUR):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset):
        return dataset.withColumn(self.getOutputCol(), hour(self.getInputCol()))
