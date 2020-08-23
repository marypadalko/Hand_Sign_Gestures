from tensorflow.keras.models import model_from_json, load_model
import h5py
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class HandSignModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_file):
        self.loaded_model = h5py.File(model_file, 'r')
        # with open(model_file, "r") as mf:
        #     # loaded_model_json = json_file.read()
        #     self.loaded_model = load_model(mf)

        # load weights into the new model
        # self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return HandSignModel.EMOTIONS_LIST[np.argmax(self.preds)]
