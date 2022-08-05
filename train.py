import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GaussianNoise, \
    GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def get_model(inputs_shape):
    """
    The get_model function creates a convolutional neural network model.

    :param inputs_shape: Define the shape of the input data
    :return: A model that has the following architecture:
    """

    inputs = Input(inputs_shape)
    x = GaussianNoise(0.05)(inputs)
    x = Conv2D(8, (13, 13), padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(16, (11, 11), padding='valid', activation='relu')(x)
    x = Conv2D(32, (9, 9), padding='valid', activation='relu')(x)
    x = Conv2D(64, (7, 7), padding='valid', activation='relu')(x)
    x = Conv2D(128, (5, 5), padding='valid', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def visualize_training_results(history, history_plot_path):
    """
    The visualize_training_results function visualizes training results on tensorboard.

    :param history: Access the accuracy and loss values of the training
    :param history_plot_path: Specify the path where to save the plot
    :return: A plot of the training and validation accuracy/loss
    """

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(212)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(history_plot_path)


def training(model, x, y, history_plot_path):
    """
    The training function trains the model for one epoch on the training dataset.

    :param model: Pass the model to be trained
    :param x: Pass the training image
    :param y: Pass the training labels
    :param history_plot_path: Specify where to save the plot of the training history
    :return: The training history
    """

    model.compile(Adam(learning_rate=1e-4),
                  loss="CategoricalCrossentropy",
                  metrics="accuracy")
    history = model.fit(
        x,
        y,
        batch_size=8,
        epochs=200,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(
                filepath="./model/cnn_orange/model-{val_loss:.2f}.tf",
                monitor='val_loss',
                mode='min')
        ])
    visualize_training_results(history, history_plot_path)


if __name__ == '__main__':
    data = np.load("./dataset/dataset.npz")
    x_train, y_train = data["train_img"], data["train_label"]
    # print(x_train, y_train)
    print(x_train.shape, y_train.shape)

    training(get_model(x_train.shape[1:]),
             x_train / 255.,
             y_train,
             history_plot_path="./model/cnn_orange/history.png")


