import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
np.set_printoptions(threshold=sys.maxsize)

def change_color(name, color_angle):
    mask = cv2.imread('segmentation/'+name)
    image = cv2.imread('dataset/'+name)

    # convert to HSV:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #mul_mask = np.ones(mask.shape)
    print(image_hsv[0, 0])
    #image_hsv[mask[:,:,1]==128] += np.array([angle_rotate, 0, 0],dtype=np.uint8)

    image_hsv[mask[:,:,1]==128, 0] = color_angle
    new_image =cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    return new_image

def change_color_tf_rand(image, mask):
    image_new = tf.image.random_hue(image, 0.5, 456)
    image = tf.where(mask == 128, image_new, image)
    #print(image)
    return image


def change_color_tf(image, mask, val):
    # image_new = (tf.image.adjust_hue(image, val) - image)
    
    # image_new = tf.image.adjust_brightness(image_new, -0.5)
    image_new = tf.cast(image, tf.float32)
    #image_new /= 255.    
    h,s,v = tf.split(tf.image.rgb_to_hsv(image_new), num_or_size_splits=3, axis=-1)
    h = tf.zeros(h.shape) + val
    image_new = tf.stack([h,s,v], axis=-1)
    image_new = tf.reshape(image_new, image.shape)
    image_new = tf.cast(tf.image.hsv_to_rgb(image_new),tf.uint8)
    #image_new = tf.image.adjust_saturation(image_new, 3.)
    image = tf.where(tf.expand_dims(tf.reduce_any(mask == 128, axis=-1),-1), image_new, image)
    return image

def make_image_white(image, mask):
    image_new = tf.image.adjust_saturation(image, 0.5)
    image = tf.where(tf.expand_dims(tf.reduce_any(mask == 128, axis=-1),-1), image_new, image)
    return image

def create_color_classes(image, mask):
    #green
    im_green = change_color_tf(image, mask, 1/3)
    #red
    im_red = change_color_tf(image, mask, 0)
    #blue
    im_blue = change_color_tf(image, mask, 2/3)
    #white
    im_white = make_image_white(image, mask)

    return im_green, im_blue, im_red, im_white

if 0: 
    names = ['2021-04-19-17-38-12-658915_3', '2021-04-19-16-21-00-350307_3','2021-03-24-11-51-15-731895_2', '2021-04-07-15-48-55-846751_2','2021-04-04-15-10-35-644558_3', '2021-04-07-15-45-40-113154_2', '2021-04-07-14-58-00-073706_3']
    for name in names:
        mask = tf.io.read_file('segmentation/'+name+'.png')
        image = tf.io.read_file('dataset/'+name+'.png')
        image = tf.image.decode_jpeg(image, channels=3)
        mask = tf.image.decode_jpeg(mask, channels=3)
        im_green, im_blue, im_red, im_white = create_color_classes(image, mask)
        fig, ax = plt.subplots(2,3)

        ax[0,0].imshow(image.numpy())
        ax[0,0].title.set_text("original")

        ax[0,1].imshow(im_green.numpy())
        ax[0,1].title.set_text("green")

        ax[0,2].imshow(im_blue.numpy())
        ax[0,2].title.set_text("blue")

        ax[1,0].imshow(im_red.numpy())
        ax[1,0].title.set_text("red")

        ax[1,2].imshow(im_white.numpy())
        ax[1,2].title.set_text("white")

        plt.show()
if 0:
    mask = tf.io.read_file('segmentation/'+'2021-03-08-13-20-33-292528_2.png')
    image = tf.io.read_file('dataset/'+'2021-03-08-13-20-33-292528_2.png')
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    # images = tf.image.resize(images, (1280, 720),
    #                        method='nearest', antialias=True)
    mask = tf.image.decode_jpeg(mask, channels=3)
    # mask = tf.image.resize(mask, (1280, 720),
    #                        method='nearest', antialias=True)
    #image = tf.image.rgb_to_hsv(image)
    # image_np_old = image.numpy()
    # image_new = change_color_tf(image, mask, 0.75)
    

    
    # image_np_1 = image_new.numpy()
    # image_new = change_color_tf(image, mask, .8)
    # image_np_2 = image_new.numpy()
    # image_new = change_color_tf(image, mask, 0.85)
    # image_np_3 = image_new.numpy()
    num = 11
    step_size = 0.1
    fig, ax = plt.subplots(num,3)
    for x in range(num):
        new, fin = change_color_tf(image, mask, x*step_size)
        ax[x,0].imshow(new.numpy())
        ax[x,1].imshow(fin.numpy())
        ax[x,0].title.set_text("hue: "+str(x*step_size))
    # ax[0,0].imshow(image_np_old)
    # ax[0,1].imshow(image_np_1)
    # ax[1,0].imshow(image_np_2)
    # ax[1,1].imshow(image_np_3)
    # image_yellow = change_color('2021-03-08-10-29-52-962403_1.png', 0)
    # image_red = change_color('2021-03-08-10-29-52-962403_1.png', 90)
    # image_green = change_color('2021-03-08-10-29-52-962403_1.png', 180)
    # image_orig = change_color('2021-03-08-10-29-52-962403_1.png', 270)
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(cv2.cvtColor(cv2.imread('dataset/2021-03-08-10-29-52-962403_1.png'),cv2.COLOR_BGR2RGB))
    # ax[0,1].imshow(image_green)
    # ax[1,0].imshow(image_red)
    # ax[1,1].imshow(image_yellow)
    # plt.show()


    # image_yellow = change_color('2021-03-08-10-35-37-357451_3.png',20)
    # image_red = change_color('2021-03-08-10-35-37-357451_3.png',90)
    # image_green = change_color('2021-03-08-10-35-37-357451_3.png',180)
    # image_orig = change_color('2021-03-08-10-35-37-357451_3.png',250)
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(image_orig)
    # ax[0,1].imshow(image_green)
    # ax[1,0].imshow(image_red)
    # ax[1,1].imshow(image_yellow)
    num = 11
    step_size = 18
    #fig, ax = plt.subplots(num)
    for x in range(num):
        ax[x,2].imshow(change_color('2021-03-08-13-20-33-292528_2.png', x*step_size))
        ax[x,2].title.set_text("hue: "+str(x*step_size ))
    plt.show()