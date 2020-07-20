import keras
import numpy as np


# 64*64 grayscale
def emotion_recog(img):
    f = open('fer2013_big_XCEPTION.54-0.66.config')
        emotions = f.readline().split('\t')

    img = [img]
    img = np.array(img)
    img = img[..., np.newaxis]
    #   https://github.com/oarriaga/face_classification
    model = keras.models.load_model('fer2013_big_XCEPTION.54-0.66.hdf5', compile=False)
    result = model.predict(img)
    result = np.argmax(result, axis=1)

    return emotions[result[0]]
