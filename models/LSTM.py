import tensorflow as tf


def build_LSTM_model(input_shape, output_num):
    inputs= tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.LSTM(100, return_sequences=True, activation='tanh')(inputs)
    x = tf.keras.layers.LSTM(50, return_sequences=False, activation='tanh')(x)

    x = tf.keras.layers.Dense(4, activation='linear')(x)
    outputs = tf.keras.layers.Dense(output_num, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, name='Adam'
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    model.summary()
    return model
