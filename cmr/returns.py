import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from keras.layers import Dense, LSTM
from keras.models import Sequential
from lazy import lazy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import timeseries_dataset_from_array  # noqa

from .data_loader import load_features


class TaReturnsForecast(object):

    def __init__(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp, resample_rule: str = '1H') -> None:
        """

        :param symbols: universe
        :param start: start timestamp
        :param end: end timestamp
        :param resample_rule：frequency
        """

        self.symbols = symbols
        self.start = start
        self.end = end
        self.resample_rule = resample_rule

    @lazy
    def alpha_source(self):
        feat = load_features(self.symbols, self.start, self.end, self.resample_rule)
        feat = feat.reorder_levels(['time', 'symbol']).sort_index()
        return feat

    @lazy
    def test_start(self):
        return ((self.end - self.start) * .9 + self.start).round('D')

    @lazy
    def test_end(self):
        return ((self.end - self.start) * .95 + self.start).round('D')

    @lazy
    def window(self):
        return 3

    @lazy
    def raw_train_test(self):
        feat = self.alpha_source
        return (
            feat.loc[feat.index.get_level_values(0) < self.test_start],
            feat.loc[(feat.index.get_level_values(0) >= self.test_start - relativedelta(days=self.window))]
        )

    @lazy
    def is_pca(self):
        return False

    @lazy
    def scaled_train_test(self):
        raw_train, raw_test = self.raw_train_test

        # standard
        standard_scaler = StandardScaler()
        scaled_train = pd.DataFrame(index=raw_train.index, data=standard_scaler.fit_transform(raw_train.values),
                                    columns=raw_train.columns)
        scaled_test = pd.DataFrame(index=raw_test.index, data=standard_scaler.transform(raw_test.values),
                                   columns=raw_test.columns)
        scaled_train['ret'] = raw_train['ret']
        scaled_test['ret'] = raw_test['ret']

        # symbol one hot encoder
        dummy_train = pd.get_dummies(scaled_train.index.get_level_values('symbol'))
        dummy_train.index = scaled_train.index
        train = pd.concat([scaled_train, dummy_train], axis=1)

        dummy_test = pd.get_dummies(scaled_test.index.get_level_values('symbol'))
        dummy_test.index = scaled_test.index
        test = pd.concat([scaled_test, dummy_test], axis=1)

        # features PCA, from 85 features (to 21 features)
        if self.is_pca:
            pca_scaler = PCA(n_components=0.90)
            train = pd.DataFrame(index=train.index, data=pca_scaler.fit_transform(train))
            test = pd.DataFrame(index=test.index, data=pca_scaler.transform(test))

        return train, test

    @lazy
    def generator_train_test(self):
        raw_train, raw_test = self.raw_train_test
        train, test = self.scaled_train_test

        # number_of_features = train.shape[1]
        train_index = pd.DataFrame()
        test_index = pd.DataFrame()

        for idx, symbol in enumerate(self.symbols):

            x_train = train.loc[(slice(None), symbol),]
            y_train = raw_train.loc[(slice(None), symbol), ['ret']][self.window:]

            x_test = test.loc[(slice(None), symbol),]
            y_test = raw_test.loc[(slice(None), symbol), ['ret']][self.window:]

            _generator_train = timeseries_dataset_from_array(x_train, y_train, sequence_length=self.window,
                                                             batch_size=None)
            _generator_test = timeseries_dataset_from_array(x_test, y_test, sequence_length=self.window,
                                                            batch_size=None)
            if idx == 0:
                generator_train = _generator_train
                generator_test = _generator_test
            else:
                generator_train = generator_train.concatenate(_generator_train)
                generator_test = generator_test.concatenate(_generator_test)
            train_index = train_index.append(y_train)
            test_index = test_index.append(y_test)

        generator_train = generator_train.batch(128)
        generator_test = generator_test.batch(128)

        return generator_train, generator_test, train_index, test_index

    @lazy
    def model(self):
        model = Sequential()
        model.add(
            LSTM(64, return_sequences=True, input_shape=(self.window, self.scaled_train_test[0].shape[1]),
                 activation="tanh", dropout=0.2))
        model.add(LSTM(64, return_sequences=False, activation="tanh"))
        model.add(Dense(32, activation="tanh"))
        model.add(Dense(1, activation="linear"))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['MSE', 'MAE'])
        return model

    def fit_model(self):
        # Train the model
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        self.model.fit(self.generator_train_test[0], epochs=7, callbacks=[callback])
        # self.model.save('universe_%sh_7epoch_tanhDualLSTM.keras' % self.window)

        # Predict
        train_result = self.generator_train_test[2].copy()
        train_result['y_train'] = self.model.predict(self.generator_train_test[0])

        test_result = self.generator_train_test[3].copy()
        test_result['y_pred'] = self.model.predict(self.generator_train_test[1])

        return train_result, test_result
