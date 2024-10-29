import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
from feature_engineering import extract_features


def main():
    # Загрузка тестовых данных и обработка
    test = pq.read_table('test.parquet').to_pandas()
    test_ids = test['id']
    test_features = extract_features(test)

    # Заполнение пропущенных значений (должен совпадать с тренировочным)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(np.load('train_features.npy'))
    X_test_features = imputer.transform(test_features)

    # Загрузка обученной модели
    rec_sys = load_model('models/rec_sys.keras')

    # Предсказание вероятностей
    y_pred_proba_test = rec_sys.predict(X_test_features).ravel()

    # Сохранение в submission.csv
    output_df = pd.DataFrame({'id': test_ids, 'score': y_pred_proba_test})
    output_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
