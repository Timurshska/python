import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from scipy.fft import fft
from scipy.signal import welch
import pywt
from sklearn.impute import SimpleImputer
import os


def extract_features(data):
    features = []
    for ts in data['values']:
        ts_series = pd.Series(ts)

        # Базовые статистические признаки
        quantiles = ts_series.quantile([0.25, 0.5, 0.75])
        basic_features = [
            np.mean(ts),
            float(stats.mode(ts)[0]),
            np.std(ts),
            np.max(ts),
            np.min(ts),
            quantiles[0.25],
            quantiles[0.5],
            quantiles[0.75],
            np.sum(np.abs(np.diff(ts))),
        ]

        # Коэффициенты Фурье (10 первых модулей)
        fft_coeffs = np.abs(fft(ts))[:10]

        # Вейвлет-преобразование на нескольких уровнях
        wavelet = 'db1'
        coeffs = pywt.wavedec(ts, wavelet, level=3)
        wavelet_coeffs = np.concatenate([c[:5] for c in coeffs if len(c) >= 5])

        # Энергетические характеристики
        energy_total = np.sum(np.square(ts))
        energy_first_half = np.sum(np.square(ts[:len(ts) // 2]))
        energy_second_half = np.sum(np.square(ts[len(ts) // 2:]))

        # Спектральная плотность
        f, psd = welch(ts, nperseg=len(ts))
        spectral_density = psd[:5]

        # Объединяем все признаки
        features.append(basic_features + list(fft_coeffs) + list(wavelet_coeffs) +
                        [energy_total, energy_first_half, energy_second_half] +
                        list(spectral_density))

    return np.array(features)


def main():
    # Загрузка и обработка данных
    train = pq.read_table('train.parquet').to_pandas()
    test = pq.read_table('test.parquet').to_pandas()

    train_features = extract_features(train)
    test_features = extract_features(test)

    # Заполнение пропущенных значений
    imputer = SimpleImputer(strategy='mean')
    train_features = imputer.fit_transform(train_features)
    test_features = imputer.transform(test_features)

    # Сохранение обработанных данных
    np.save('train_features.npy', train_features)
    np.save('test_features.npy', test_features)
    np.save('label.npy', train['label'].values)


if __name__ == '__main__':
    main()
