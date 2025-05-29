import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
from matplotlib import pyplot as plt

#done with image from path, can be changed according to needs
def pad_with_tf(image_path, target_height, target_width):
    raw_img = tf.io.read_file(image_path, 'rb')
    img = tf.image.decode_png(raw_img, channels=3)
    image = ops.convert_to_tensor(img, name='image')
    padded_img = tf.image.resize_with_crop_or_pad(image, target_height, target_width)
    return(padded_img)


if __name__ == '__main__':
    #1240 768
    #original 1280x720 zu 768 auffüllen
    padded = pad_with_tf('test.png', 768,1280)
    plt.imshow(padded)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    print(padded)