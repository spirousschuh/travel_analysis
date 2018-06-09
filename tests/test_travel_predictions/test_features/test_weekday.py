from travel_analysis.const import WEEKDAY, DAY_VECTOR
from travel_analysis.travel_predictions.features.create_all import \
    get_features_pipeline
from travel_analysis.travel_predictions.features.weekday import Weekday


def test_get_weekdays(training_sample):
    # given
    feature_creator = Weekday()

    # when
    df_with_feature = feature_creator.transform(training_sample)

    # then
    weekdays = [row[WEEKDAY] for row in df_with_feature.collect()]
    assert weekdays == [5, 6, 7, 1, 1, 2, 2, 3, 4, 4]


def test_one_hot_encoding_of_the_weekdays(training_sample):
    # given
    features_pipeline = get_features_pipeline()
    trained_feature_pipeline = features_pipeline.fit(training_sample)

    # when
    df_with_one_hot = trained_feature_pipeline.transform(training_sample)

    # then
    vector_encodings = [row[DAY_VECTOR].indices[0]
                        for row in df_with_one_hot.collect()]
    assert vector_encodings == [5, 6, 7, 1, 1, 2, 2, 3, 4, 4]
