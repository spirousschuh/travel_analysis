import pyspark
SPARK_APP_NAME = 'travel_analysis'

def get_spark_session(app_name=SPARK_APP_NAME):
    # configure
    conf = pyspark.SparkConf()
    conf.set('spark.app.name', app_name)
    conf.set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
    conf.set('spark.debug.maxToStringFields', 1000)

    # init & return
    sc = pyspark.SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel('WARN')
    return pyspark.SQLContext(sparkContext=sc)


def get_dataframe(data_path):
    spark_session = get_spark_session()
    # get raw training data
    return spark_session.read.csv(
        data_path, header=True, inferSchema=True)
