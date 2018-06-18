import datetime

import pytest
from pyspark import Row

from travel_analysis.const import NINE, NINE_PRE
from travel_analysis.travel_predictions.model.linear_regression import train
from travel_analysis.utils.io_spark import get_spark_session


@pytest.fixture
def spark_session():
    return get_spark_session()


@pytest.fixture
def training_sample(spark_session):
    rows = [
        Row(index=0, ride_departure=datetime.datetime(2015, 1, 1, 8, 15),
            capacity=82.0, tickets_9_eur=21.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='B->A'),
        Row(index=1, ride_departure=datetime.datetime(2015, 1, 2, 9, 15),
            capacity=82.0, tickets_9_eur=12.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='A->B'),
        Row(index=2, ride_departure=datetime.datetime(2015, 1, 3, 10, 15),
            capacity=82.0, tickets_9_eur=33.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='B->A'),
        Row(index=3, ride_departure=datetime.datetime(2015, 1, 4, 11, 45),
            capacity=82.0, tickets_9_eur=25.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='A->B'),
        Row(index=4, ride_departure=datetime.datetime(2015, 1, 4, 12, 45),
            capacity=82.0, tickets_9_eur=32.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='B->A'),
        Row(index=5, ride_departure=datetime.datetime(2015, 1, 5, 13, 45),
            capacity=82.0, tickets_9_eur=54.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='A->B'),
        Row(index=6, ride_departure=datetime.datetime(2015, 1, 5, 15, 15),
            capacity=82.0, tickets_9_eur=59.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='B->A'),
        Row(index=7, ride_departure=datetime.datetime(2015, 1, 6, 16, 15),
            capacity=82.0, tickets_9_eur=77.0, tickets_12_eur=1.0,
            tickets_15_eur=1.0, tickets_19_eur=2.0, direction='A->B'),
        Row(index=8, ride_departure=datetime.datetime(2015, 1, 7, 17, 15),
            capacity=82.0, tickets_9_eur=63.0, tickets_12_eur=0.0,
            tickets_15_eur=0.0, tickets_19_eur=0.0, direction='B->A'),
        Row(index=9, ride_departure=datetime.datetime(2015, 1, 7, 18, 45),
            capacity=82.0, tickets_9_eur=78.0, tickets_12_eur=2.0,
            tickets_15_eur=1.0, tickets_19_eur=2.0, direction='A->B')]

    return spark_session.createDataFrame(rows)


@pytest.fixture
def missing_sample(spark_session):
    rows = [Row(index=1, ride_departure=datetime.datetime(2015, 1, 7, 17, 15),
                capacity=100.0, tickets_9_eur=None, tickets_12_eur=10.0,
                tickets_15_eur=5.0, tickets_19_eur=5.0, direction='B->A'),
            Row(index=1, ride_departure=datetime.datetime(2015, 1, 7, 18, 45),
                capacity=100.0, tickets_9_eur=80.0, tickets_12_eur=10.0,
                tickets_15_eur=5.0, tickets_19_eur=5.0, direction='A->B')]

    return spark_session.createDataFrame(rows)


@pytest.fixture
def sample_predictions(training_sample):
    trained_model = train(training_sample, NINE, NINE_PRE)
    return trained_model.transform(training_sample)
