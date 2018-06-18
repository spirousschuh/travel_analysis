from travel_analysis.const import HOUR
from travel_analysis.travel_predictions.features.Hour import Hour


def test_get_hours(training_sample):
    # given
    feature_creator = Hour()

    # when
    df_with_feature = feature_creator.transform(training_sample)

    # then
    hours = [row[HOUR] for row in df_with_feature.collect()]
    assert hours == [8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
