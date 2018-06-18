from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


class Fraction(Transformer, HasInputCols, HasOutputCol):
    def __init__(self, inputCols, outputCol):
        super().__init__()
        self._setDefault(inputCols=inputCols, outputCol=outputCol)
        self._set(inputCols=inputCols, outputCol=outputCol)

    def _single_fraction(self, first, second):
        return first / second

    def _transform(self, dataset):
        spark_devision = udf(self._single_fraction, FloatType())
        return dataset.withColumn(self.getOutputCol(),
                                  spark_devision(*self.getInputCols()))
