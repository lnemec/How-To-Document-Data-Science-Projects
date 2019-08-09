"""Functions for downloading and reading MNIST data."""

import numpy as np

def load():
    """
    **Dataset**

    The MNIST dataset was created by combining the samples of the original
    datasets from National Institute of Standards and Technology (NIST). The NIST
    training data was taken from American Census Bureau employees. The original
    NIST dataset contained black and white images. Before including them into the
    MNIST dataset, they were normalized to fit into a 28x28 pixel bounding box and
    anti-aliased, which introduced grayscale levels. The other half of the dataset
    contains handwritten digits from American high school students. The final MNIST
    database contains 60,000 training images and 10,000 testing images.

    It is a good database for people who want to try machine learning techniques
    and pattern recognition methods on real-world data while spending minimal
    efforts on preprocessing and formatting.

    The MNIST dataset is distributed with the TensorFlow code.

    :return: data dictionary
    """
    from tensorflow.keras.datasets.mnist import load_data
    from tensorflow.keras.utils import to_categorical

    data = dict()

    (x_train, y_train), (x_test, y_test) = load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_test = to_categorical(y_test, num_classes=10, dtype='int')
    y_train = to_categorical(y_train, num_classes=10, dtype='int')

    data = __split_validation(x_train, y_train, data)

    data['x_test'] = x_test
    data['y_test'] = y_test

    data = __flatten_arrays( data, ['x_test', 'x_train', 'x_validation'] )

    data = __dataset_size(data)

    return data

def __split_validation(x_train : np.ndarray, y_train : np.ndarray, data : dict ):

    data['x_validation'] = x_train[55000:]
    data['y_validation'] = y_train[55000:]
    data['x_train'] = x_train[0:55000]
    data['y_train'] = y_train[0:55000]

    return data


def __dataset_size(data):

    data.update({'n_train': data['x_train'].shape[0]})
    data.update({'n_test': data['x_test'].shape[0]})
    data.update({'n_validation': data['x_validation'].shape[0]})
    data.update({'image_size' : data['x_train'].shape[1]})
    data.update({'labels_size' : data['y_train'].shape[1]})

    return data

def __flatten_arrays(data: dict, l_keys: list) -> dict:

    for k in l_keys:
        data[k] = data[k].reshape((data[k].shape[0], -1))

    return data
