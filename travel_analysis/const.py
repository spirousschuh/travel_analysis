# original features
INDEX = 'index'
DIRECTION = 'direction'
NINETEEN = 'tickets_19_eur'
FIFTEEN = 'tickets_15_eur'
TWELF = 'tickets_12_eur'
NINE = 'tickets_9_eur'
CAPACITY = 'capacity'
DEPARTURE = 'ride_departure'

# features created
WEEKDAY = '_weekday'
HOUR = '_hour'
DAY_VECTOR = 'weekday_vector'
HOLIDAY = 'holiday'
BEFORE_HOLIDAY = 'before_holiday'
DIRECTION_FLAG = 'direction_flag'
RIDES_PER_DAY = 'rides_per_day'

FEATURES = 'features'
PURE_FEATURES = 'pure_features'

NINETEEN_FEAT = '19_eur_features'
FIFTEEN_FEAT = '15_eur_features'
TWELF_FEAT = '12_eur_features'

NINETEEN_PRE = 'tickets_19_eur_prediction'
FIFTEEN_PRE = 'tickets_15_eur_prediction'
TWELF_PRE = 'tickets_12_eur_prediction'
NINE_PRE = 'nice_euro_prediction'

AVG_9 = 'rolling_avg_9_euro'
AVG_12 = 'rolling_avg_12_euro'
AVG_15 = 'rolling_avg_15_euro'
AVG_19 = 'rolling_avg_19_euro'

FRAC_9 = 'fraction_9'
FRAC_12 = 'fraction_12'
FRAC_15 = 'fraction_15'
FRAC_19 = 'fraction_19'

ROLLING_AVGS = [AVG_9, AVG_12, AVG_15, AVG_19]

FULL_TRAINING_DATA = '/home/chlange/Documents/Bewerbung/2018 zweiter Job/' \
                     'Flixbus/Aufgabe/travel_analysis/data/training_data.csv'
FULL_TRAIN_DATA_FEATURES = '/home/chlange/Documents/Bewerbung/2018 zweiter Job/' \
                     'Flixbus/Aufgabe/travel_analysis/data/training_data_features.csv'
DIRECTION_IDX = 'direction_index'
LABEL_COLS = [NINE, TWELF, FIFTEEN, NINETEEN]
