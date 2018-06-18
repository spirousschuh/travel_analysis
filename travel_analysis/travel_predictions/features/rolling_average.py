from functools import partial
from operator import itemgetter

from pyparsing import col
from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Params, \
    Param, \
    TypeConverters
from pyspark.sql import Window
from pyspark.sql.functions import avg, udf
from pyspark.sql.types import StringType, DateType, FloatType


class RollingAverageParams(HasInputCols, HasOutputCols):
    _date_column = Param(Params._dummy(), "_date_column",
                         "name of the date column",
                         typeConverter=TypeConverters.toString)
    _first_day_of_window = Param(Params._dummy(), "_first_day_of_window",
                                 "first day to consider for the rolling "
                                 "average",
                                 typeConverter=TypeConverters.toInt)
    _time_span_avg = Param(Params._dummy(),
                           "_time_span_avg",
                           "number of days in the rolling avg",
                           typeConverter=TypeConverters.toInt)

    @property
    def date_column(self):
        return self.getOrDefault(self._date_column)

    @property
    def first_day_of_window(self):
        return self.getOrDefault(self._first_day_of_window)

    @property
    def time_span_avg(self):
        return self.getOrDefault(self._time_span_avg)


class RollingAverageEstimator(Estimator, RollingAverageParams):
    def __init__(self,
                 inputCols,
                 outputCols,
                 date_column,
                 first_day_of_window,
                 time_span_avg,
                 ):
        super().__init__()
        self._setDefault(inputCols=inputCols,
                         outputCols=outputCols,
                         _date_column=date_column,
                         _first_day_of_window=first_day_of_window,
                         _time_span_avg=time_span_avg)
        self._set(inputCols=inputCols,
                  outputCols=outputCols,
                  _date_column=date_column,
                  _first_day_of_window=first_day_of_window,
                  _time_span_avg=time_span_avg)

    def get_rolling_avgs_per_date(self, data):
        date_col = 'date_col'
        data = data.withColumn(date_col,
                               data[self.date_column].cast(DateType()))
        filtered = data.select(date_col, *self.getInputCols())
        grouped_by_date = filtered.groupby(date_col).avg()

        first = - self.first_day_of_window
        last = - self.first_day_of_window + (self.time_span_avg - 1)
        avg_window = Window.orderBy(date_col).rowsBetween(
            first, last)

        avg_per_day_cols = ['avg({col})'.format(col=in_col)
                            for in_col in self.getInputCols()]

        for in_col, out_col in zip(avg_per_day_cols, self.getOutputCols()):
            grouped_by_date = grouped_by_date.withColumn(out_col,
                                                         avg(in_col).over(
                                                             avg_window))
        rolling_avgs = grouped_by_date.select(date_col, *self.getOutputCols())
        date = itemgetter(date_col)
        avgs = itemgetter(*self.getOutputCols())
        return {date(row): list(avgs(row)) for row in rolling_avgs.collect()}

    def _fit(self, data):
        # group by the pure date of the ride
        avgs_per_date = self.get_rolling_avgs_per_date(data)
        return RollingAverage(avgs_per_date=avgs_per_date,
                              inputCols=self.getInputCols(),
                              outputCols=self.getOutputCols(),
                              date_column=self.date_column,
                              )


class RollingAverage(Transformer, RollingAverageParams):
    def __init__(self,
                 avgs_per_date,
                 inputCols,
                 outputCols,
                 date_column,
                 ):
        super().__init__()
        self._setDefault(inputCols=inputCols,
                         outputCols=outputCols,
                         _date_column=date_column,
                         )
        self._set(inputCols=inputCols,
                  outputCols=outputCols,
                  _date_column=date_column,
                  )
        self.avgs_per_date = avgs_per_date

    def _single_avg(self, datetime, idx):
        avg = self.avgs_per_date[datetime.date()][idx]
        if avg is None:
            avg = 0.
        return avg

    def _transform(self, features_df):
        spark_maps = [
            (udf(partial(self._single_avg, idx=idx), FloatType()), column)
            for (idx, column) in enumerate(self.getOutputCols())]
        for spark_map, output_column in spark_maps:
            features_df = features_df.withColumn(output_column,
                                                 spark_map(self.date_column))
        return features_df
