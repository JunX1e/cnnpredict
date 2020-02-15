# v3 u can draw pic and predict it
# tf v2.x need pil to cut pic
import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN


'''
python 3.7
tensorflow 2.0.0b0
pillow(PIL) 7.0.0
'''


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('../ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        flatten_img = np.reshape(img, (28, 28, 1))
        x = np.array([1 - flatten_img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('        -> Predict digit:', np.argmax(y[0]))


# change the size


def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


if __name__ == "__main__":
    app = Predict()
    file_in = 'e:/testpic/3.png'
    img = Image.open(file_in)
    if img.size != (28, 28):
        width = 28
        height = 28
        file_out = 'e:/testpic/3.png'
        produceImage(file_in, width, height, file_out)
        print("pic size changed")
        app.predict('e:/testpic/3.png')
    else:
        app.predict('e:/testpic/3.png')



