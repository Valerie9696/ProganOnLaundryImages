from config import *
import change_colors
import padding
from util import *


def normalize(image):
    '''
        normalizing the images to [-1, 1]
    '''
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image

def augmentation(image):
    '''
        Perform some augmentation
    '''
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image(file_path, target_size=image_size, target_size2=image_width):
    images = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.resize(images, (target_size, target_size2),
                           method='nearest', antialias=True)
    images = augmentation(images)
    images = normalize(images)
    return images

def generate_color_dataset(target_size, target_size2, batch_size):
    list_ds = tf.data.Dataset.list_files(SEGMENTATION_DIR+ '/*').map(lambda x: tf.strings.substr(x, 15, 32))
    global total_data_number 
    total_data_number = len(os.listdir(SEGMENTATION_DIR))
    preprocess_function = partial(change_color_random, target_size=target_size, target_size2=target_size2)
    train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)
    print('list_sds: ',next(iter(list_ds)).numpy())
    return train_data

def generate_4_color_dataset(target_size, target_size2, batch_size):
    list_ds = tf.data.Dataset.list_files(SEGMENTATION_DIR+ '/*').map(lambda x: tf.strings.substr(x, 15, 32))
    global total_data_number 
    total_data_number = len(os.listdir(SEGMENTATION_DIR))*4
    preprocess_function = partial(get_4_color_im, target_size=target_size, target_width=target_size2)
    train_data = list_ds.map(preprocess_function)
    train_data = train_data.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x)).shuffle(100).batch(batch_size)
    for element in train_data:
        print(element.shape)
        break
    print(train_data.cardinality().numpy())
    return train_data 

def get_4_color_im(file_path, target_size=512, target_width=512):
    images = padding.pad_with_tf(DATA_BASE_DIR + '/' + file_path, 768, 1280)
    masks = padding.pad_with_tf(SEGMENTATION_DIR + '/' + file_path, 768, 1280)
    g,b,r,w = change_colors.create_color_classes(images, masks)
    g,b,r,w =   tf.image.resize(g, (target_size, target_width),method='nearest', antialias=True), tf.image.resize(b, (target_size, target_width),method='nearest', antialias=True),tf.image.resize(r, (target_size, target_width),method='nearest', antialias=True),tf.image.resize(w, (target_size, target_width),method='nearest', antialias=True)
    g,b,r,w = augmentation(g),augmentation(b),augmentation(r),augmentation(w)
    g,b,r,w = normalize(g),normalize(b),normalize(r),normalize(w)
    images = tf.stack([g,b,r,w])
    return images
def change_color_random(file_path, target_size=512, target_size2=512):

    images = padding.pad_with_tf(DATA_BASE_DIR + '/' + file_path, 768, 1280)
    masks = padding.pad_with_tf(SEGMENTATION_DIR + '/' + file_path, 768, 1280)
    # convert the compressed string to a 3D uint8 tensor
    # images = tf.image.decode_jpeg(images, channels=3)
    # masks = tf.image.decode_jpeg(masks, channels=3)
    images = change_colors.change_color_tf_rand(images, masks)
    
    return images

def generate_dataset(target_size, target_size2, batch_size):
    list_ds = tf.data.Dataset.list_files(DATA_BASE_DIR + '/*')

    for f in list_ds.take(5):
        print(f.numpy())
    preprocess_function = partial(preprocess_image, target_size=target_size, target_size2=target_size2)
    train_data = list_ds.map(preprocess_function).shuffle(100).batch(batch_size)
    return train_data


#kernel_initializer = RandomNormal(mean=0.0, stddev=1.0)
kernel_initializer = 'he_normal'

class PixelNormalization(tf.keras.layers.Layer):
    """
    Arguments:
      epsilon: a float-point number, the default is 1e-8
    """
    def __init__(self, epsilon=1e-8):
        super(PixelNormalization, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
    
    def compute_output_shape(self, input_shape):
        return input_shape


def upsample_block(x, filters1, filters2, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        Upsampling + 2 Convolution-Activation
    '''
    upsample = UpSampling2D(size=2, interpolation='nearest')(x)
    upsample_x = EqualizeLearningRate(Conv2D(filters1, kernel_size, strides, padding=padding,
                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(upsample)
    x = PixelNormalization()(upsample_x)
    x = Activation(activation)(x)
    x = EqualizeLearningRate(Conv2D(filters2, kernel_size, strides, padding=padding,
                                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = PixelNormalization()(x)
    x = Activation(activation)(x)
    return x, upsample

def downsample_block(x, filters1, filters2, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        2 Convolution-Activation + Downsampling
    '''
    x = EqualizeLearningRate(Conv2D(filters1, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(x)
    x = Activation(activation)(x)
    x = EqualizeLearningRate(Conv2D(filters2, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = Activation(activation)(x)
    downsample = AveragePooling2D(pool_size=2)(x)

    return downsample


# output_activation = tf.keras.activations.linear
output_activation = tf.keras.activations.tanh

