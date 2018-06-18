from travel_analysis.const import NINE, CAPACITY, FRAC_9
from travel_analysis.travel_predictions.features.fraction import Fraction


def test_fraction_transformation(training_sample):
    # given
    freature_creator = Fraction(inputCols=[NINE, CAPACITY],
                                outputCol=FRAC_9)

    # when
    df_with_fraction = freature_creator.transform(training_sample)

    # then
    fractions = [row[FRAC_9] for row in df_with_fraction.collect()]
    assert fractions
