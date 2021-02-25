from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib


DEFAULT_WEIGHTS_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


def download_weights(url=DEFAULT_WEIGHTS_URL, target="inception_v3.h5"):
    urllib.request.urlretrieve(url, target)


def build_classifier(weights=None):
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)
    if weights is not None:
        pre_trained_model.load_weights(weights)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    print(pre_trained_model.summary())

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)
    return model
