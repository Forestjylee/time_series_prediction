import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
import matplotlib.pyplot as plt
from  models.LSTM import build_LSTM_model
from data_loader import get_train_and_test_data


target_filepath = f"lstm_model.h5"


def iterative_predict(model, first_window: np.ndarray, amount: int):
    model.load_weights(target_filepath)
    if isinstance(first_window, np.ndarray):
        first_window = first_window.tolist()
    window = deepcopy(first_window)
    predict_res = []
    for _ in tqdm(range(amount)):
        predict_res.append(model.predict(
            np.array([window]).astype("float64"), verbose=False
        )[0])
        window.append(predict_res[-1])
        window = window[1:]
    return predict_res


def model_evaluate(model, X_test, y_test):
    model.load_weights(target_filepath)
    y_predict = model.predict(X_test)
    # y_predict = iterative_predict(model, X_test[0, :], len(y_test))
    plt.plot(y_predict, 'r')
    plt.plot(y_test, 'g-')
    plt.title('This pic is drawed using Standard Data')
    plt.legend(['predict', 'true'])
    plt.show()


def plot_history(history_df):
    # 画训练曲线
    def plot_learning_curves(history_df):
        history_df.plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    plot_learning_curves(history_df)


def train(model, epochs, X_train, y_train):
    try:
        print("Training...")

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=10)
        select_best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=target_filepath, monitor='val_loss', mode='min',
            save_best_only=True, verbose=1, period=1, save_weights_only=False
        )

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128,
                            # validation_data=(X_validation, y_validation),
                            validation_split=0.1,
                            callbacks=[earlystop_callback, select_best_checkpoint])
        # pd.DataFrame(history.history).to_csv('training_log.csv', index=False)   # save train history
        # plot_history(pd.DataFrame(history.history))

        print(f"Save lstm model to {target_filepath} success")
    except KeyboardInterrupt:
        print("prediction exception")
    finally:
        return model


if __name__ == '__main__':
    model = build_LSTM_model((10, 1), 1)
    X_train, y_train, X_test, y_test = get_train_and_test_data()
    train(model, 10, X_train, y_train)
    model_evaluate(model, X_test, y_test)
