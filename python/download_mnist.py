if __name__ == "__main__":
    import tensorflow as tf
    import typing
    def save_pixels_and_labels (X, y, save: typing.TextIO):
        for pixels, label in zip (X, y):
            pixels_list = [str(num) for num in (pixels.reshape(-1, 784)/255).tolist()[0]]
            save.write("{} {}\n".format(label, " ".join(pixels_list)))

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    with open("../mnist.train.txt", "w") as filehandle:
        save_pixels_and_labels(X_train, y_train, filehandle)
    with open("../mnist.test.txt", "w") as filehandle:
        save_pixels_and_labels(X_test, y_test, filehandle)

