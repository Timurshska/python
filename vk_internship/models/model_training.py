import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import AUC

def create_nn_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])
    return model

def main():
    # Загрузка обработанных данных
    X_train = np.load('data/train_features.npy')
    y_train = np.load('data/label.npy')

    # Инициализация и обучение модели
    nn_model = create_nn_model(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

    # Сохранение обученной модели
    nn_model.save('models/rec_sys.keras')

if __name__ == '__main__':
    main()
