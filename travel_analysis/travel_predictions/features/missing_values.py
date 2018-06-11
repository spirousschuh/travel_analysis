from pyspark.ml import Estimator, Transformer
from pyspark.ml.param.shared import HasInputCols, HasInputCol, HasOutputCol, \
    Param, Params, TypeConverters
from pyspark.sql.functions import sum as spark_sum, udf
from pyspark.sql.types import FloatType

from travel_analysis.const import NINE, TWELF, FIFTEEN, NINETEEN


class MissingEstimator(Estimator, HasInputCols, HasOutputCol):
    def __init__(self, outputCol,
                 inputCols=[NINE, TWELF, FIFTEEN, NINETEEN]
                 ):
        super().__init__()
        self._setDefault(outputCol=outputCol,
                         inputCols=inputCols)
        self._set(outputCol=outputCol,
                  inputCols=inputCols)

    def get_distribution(self, dataset):
        ticket_categories = dataset.select(self.getInputCols())
        aggregation_row = ticket_categories.agg(
            *[spark_sum(col) for col in self.getInputCols()]).collect()
        sums = aggregation_row[0].asDict().values()
        total = sum(sums)
        return [one / total for one in sums]

    def _fit(self, dataset):
        distribution = self.get_distribution(dataset)
        return MissingValues(distribution,
                             self.getOutputCol(),
                             self.getInputCols())


class MissingValues(Transformer, HasInputCols, HasOutputCol):
    _distribution = Param(Params._dummy(), '_distribution',
                          "distribution of seats over ticket prices ",
                          typeConverter=TypeConverters.toListFloat)

    def __init__(self,
                 distribution,
                 outputCol,
                 inputCols=[NINE, TWELF, FIFTEEN, NINETEEN]
                 ):
        super().__init__()
        self._setDefault(_distribution=distribution,
                         outputCol=outputCol,
                         inputCols=inputCols)
        self._set(_distribution=distribution,
                  outputCol=outputCol,
                  inputCols=inputCols)

    @property
    def distribution(self):
        """
        Gets the value of the Spark parameter for the distribution.
        """
        return self.getOrDefault(self._distribution)

    def single_replacement(self, nine, twelf, fifteen, nineteen):

        idx = [i for (i, col) in enumerate(self.getInputCols())
               if col == self.getOutputCol()][0]
        values = [nine, twelf, fifteen, nineteen]

        number_of_nulls = sum([int(one is None) for one in values])
        assert number_of_nulls <= 1
        if values[idx] is None:
            coeffs = self.distribution[:idx] + self.distribution[(idx + 1):]
            passengers = values[:idx] + values[(idx + 1):]
            return 1 / sum(coeffs) * sum(passengers) * self.distribution[idx]
        else:
            return values[idx]

    def _transform(self, dataset):
        spark_replace = udf(self.single_replacement, FloatType())
        return dataset.withColumn(self.getOutputCol(),
                                  spark_replace(*self.getInputCols()))
