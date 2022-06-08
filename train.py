import tensorflow as tf
import matplotlib.pyplot as plt
from  models.LSTM import build_LSTM_model
from data_loader import get_train_and_test_data


target_filepath = f"lstm_model.h5"


def model_evaluate():
    _, _, X_test, y_test= get_train_and_test_data()
    model = build_LSTM_model((10, 1), 1)
    model.load_weights(target_filepath)
    y_predict = model.predict(X_test)
    plt.plot(y_predict, 'r')
    plt.plot(y_test, 'g-')
    plt.title('This pic is drawed using Standard Data')
    plt.legend(['predict', 'true'])
    plt.show()


def train(epochs):
    X_train, y_train, _, _= get_train_and_test_data()
    
    model = build_LSTM_model((10, 1), 1)
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
        # pd.DataFrame(history.history).to_csv('training_log.csv', index=False)

        print(f"Save lstm model to {target_filepath} success")
    except KeyboardInterrupt:
        print("prediction exception")
    finally:
        return model


if __name__ == '__main__':
    train(epochs=10)
    model_evaluate()
