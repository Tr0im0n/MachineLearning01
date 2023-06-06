
import tensorflow as tf
import numpy as np
import pandas as pd


# useless, keras has a built-in function, which you don't even need
def array_to_categorical(array: np.ndarray):
    ans = np.zeros((len(array), 7))
    for i, q in enumerate(array):
        ans[i, q-3] = 1
    return ans


def regression(df: pd.DataFrame, results: np.ndarray, *,
               hidden_layer: bool = False, epochs: int = 128) -> None | tuple[tuple[float, ...], str]:
    df_shape = df.shape
    train_ratio = 0.8
    train_samples = int(train_ratio * df_shape[0])

    training_df = df[:train_samples]
    testing_df = df[train_samples:]
    training_results = results[:train_samples]
    testing_results = results[train_samples:]

    model = tf.keras.models.Sequential()
    if not hidden_layer:
        initial_bias = tf.keras.initializers.Constant(value=5.5)
        model.add(tf.keras.layers.Dense(1, input_shape=(df_shape[1],), activation='linear',
                                        bias_initializer=initial_bias))
    else:
        model.add(tf.keras.layers.Dense(df_shape[1], input_shape=(df_shape[1],), activation='linear'))
        model.add(tf.keras.layers.Dense(1))

    print(model.summary())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_df, training_results, epochs=epochs)

    loss = model.evaluate(testing_df, testing_results)
    print("Test Loss:", loss)

    if hidden_layer:
        return
    weights, bias = model.layers[0].get_weights()
    bias_weights = float(bias), *tuple(float(i) for i in weights)
    print("Weights:")
    print(*tuple(f"{i:.4f}" for i in bias_weights[1:]), sep=", ")
    print("Bias:")
    print(f"{bias_weights[0]:.4f}")
    return bias_weights, f"{loss:.4f}"


def classification(df: pd.DataFrame, results: np.ndarray, *,
                   epochs: int = 128) -> None:

    df_shape = df.shape
    train_ratio = 0.5
    val_ratio = 0.1
    train_samples = int(train_ratio * df_shape[0])
    val_samples = int(train_ratio+val_ratio * df_shape[0])

    classification_array = results - 3

    training_df = df[:train_samples]
    training_results = classification_array[:train_samples]
    validation_df = df[train_samples:val_samples]
    validation_results = classification_array[train_samples:val_samples]
    testing_df = df[val_samples:]
    testing_results = classification_array[val_samples:]

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(df_shape[1],)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_df, training_results, epochs=epochs, batch_size=32,
              validation_data=(validation_df, validation_results))

    for i, layer in enumerate(model.layers):
        weights, biases = layer.get_weights()
        print(f"Layer {i} Weights:")
        print(weights)
        print(f"Layer {i} Biases:")
        print(biases)

    test_loss, test_acc = model.evaluate(testing_df, testing_results)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
