import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm

def model():
    model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])
    print(model.summary())
    return model


if __name__ == "__main__":
    model = model()