import os

from pyspark.sql import DataFrame

import travel_analysis
from travel_analysis.utils.io_spark import get_dataframe


def test_read_csv_input():
    # given
    base_folder = os.path.dirname(os.path.dirname(travel_analysis.__file__))
    real_training_data = os.path.join(base_folder, 'data', 'training_data.csv')

    # when
    df = get_dataframe(real_training_data)

    # then
    assert isinstance(df, DataFrame)
