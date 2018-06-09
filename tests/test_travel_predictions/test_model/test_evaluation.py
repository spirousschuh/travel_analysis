from travel_analysis.const import NINE, NINE_PREDICTION
from travel_analysis.travel_predictions.model.evaluation import \
    get_r2_metrics


def test_evaluation(sample_predictions):

    # when
    r2 = get_r2_metrics(input_df=sample_predictions,
                        label_col=NINE,
                        prediction_col=NINE_PREDICTION)

    # then
    assert r2 > 0