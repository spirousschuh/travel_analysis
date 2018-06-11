from travel_analysis.const import NINE, NINE_PRE, NINETEEN, FIFTEEN, TWELF, \
    NINETEEN_PRE, FIFTEEN_PRE, TWELF_PRE, FULL_TRAINING_DATA
from travel_analysis.travel_predictions.model.evaluation import \
    get_r2_metrics, evaluate_model_on_all_price_levels
from travel_analysis.travel_predictions.model.linear_regression import train
from travel_analysis.utils.io_spark import get_dataframe


def test_evaluation(sample_predictions):
    # when
    r2 = get_r2_metrics(input_df=sample_predictions,
                        label_col=NINE,
                        prediction_col=NINE_PRE)

    # then
    assert r2 > 0


def test_single_evaluation():
    input_features = get_dataframe(FULL_TRAINING_DATA)
    label_cols = [NINETEEN, FIFTEEN, TWELF, NINE]
    prediction_cols = [NINETEEN_PRE, FIFTEEN_PRE, TWELF_PRE, NINE_PRE]

    idx = 0
    model = train(input_features, label_cols[idx], prediction_cols[idx])

    predictions = model.transform(input_features)

    r2_metrics = get_r2_metrics(predictions, label_cols[idx],
                                prediction_cols[idx])
    print(r2_metrics)


def test_evaluate_model_on_all_price_levels(training_sample):
    # when
    metrices = evaluate_model_on_all_price_levels(training_sample)

    # then
    assert all([0 < one_r2 < 1 for one_r2 in metrices])
