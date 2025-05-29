from config import *
from preprocessing import *
from util import *

def discriminator_block(x, min_size=4):
    '''
        Discriminator output block
    '''
    x = MinibatchSTDDEV()(x)
    x = EqualizeLearningRate(Conv2D(512, 3, strides=1, padding='same',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_1')(x)
    x = LeakyReLU()(x)
    x = EqualizeLearningRate(Conv2D(512, min_size, strides=1, padding='valid',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_2')(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = EqualizeLearningRate(Dense(1, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_dense')(x)
    return x

def build_4x4_discriminator():
    '''
        4 * 4 Discriminator
    '''
    inputs = Input((4,4,3))
    # Not used in 4 * 4
    alpha = Input((1), name='input_alpha')
    # From RGB
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    x = from_rgb(inputs)
    x = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='conv2d_up_channel')(x)
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_5x3_discriminator():
    '''
        5 * 3 Discriminator
    '''
    inputs = Input((3,5,3))
    # Not used in 4 * 4
    alpha = Input((1), name='input_alpha')
    # From RGB
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(5, 3))
    x = from_rgb(inputs)
    x = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='conv2d_up_channel')(x)
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_8x8_discriminator():
    '''
        8 * 8 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((8,8,3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable block
    ########################
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_10x6_discriminator():
    '''
        10 * 6 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((6,10,3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(5, 3))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(10, 6))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable block
    ########################
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_16x16_discriminator():
    '''
        16 * 16 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((16, 16, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_20x12_discriminator():
    '''
        20 * 12 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((12,20,3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(10,6))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(20,12))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_32x32_discriminator():
    '''
        32 * 32 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((32, 32, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(16, 16))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_40x24_discriminator():
    '''
        40 * 24 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((24,40,3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(20,12))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(40,24))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(40,24))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_64x64_discriminator():
    '''
        64 * 64 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((64, 64, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(32, 32))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=256, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_80x48_discriminator():
    '''
        80 * 48 Discriminator
    '''
    fade_in_channel = 512
    inputs = Input((48, 80, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(40, 24))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(80, 48))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=256, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(80,48))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(40,24))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_128x128_discriminator():
    '''
        128 * 128 Discriminator
    '''
    fade_in_channel = 256
    inputs = Input((128, 128, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
   
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(64, 64))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=128, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_160x96_discriminator():
    '''
        160 * 96 Discriminator
    '''
    fade_in_channel = 256
    inputs = Input((96, 160, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
   
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(80, 48))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(160, 96))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=128, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(160,96))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(80,48))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(40,24))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_256x256_discriminator():
    '''
        256 * 256 Discriminator
    '''
    fade_in_channel = 128
    inputs = Input((256, 256, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(128, 128))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=64, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_320x192_discriminator():
    '''
        320 * 192 Discriminator
    '''
    fade_in_channel = 128
    inputs = Input((192, 320, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(128, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(160, 96))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(320, 192))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=64, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(320,192))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(160,96))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(80,48))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(40,24))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_512x512_discriminator():
    '''
        512 * 512 Discriminator
    '''
    fade_in_channel = 64
    inputs = Input((512, 512, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(256, 256))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(32, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(512, 512))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=32, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(512,512))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=64, filters2=128, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
    x = discriminator_block(x)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model

def build_640x384_discriminator():
    '''
        640 * 384 Discriminator
    '''
    fade_in_channel = 64
    inputs = Input((384, 640, 3))
    alpha = Input((1), name='input_alpha')
    downsample = AveragePooling2D(pool_size=2)
    
    ########################
    # Left branch in the paper
    ########################
    previous_from_rgb = EqualizeLearningRate(Conv2D(64, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(320, 192))
    l_x = previous_from_rgb(downsample(inputs))
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    from_rgb = EqualizeLearningRate(Conv2D(32, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(640, 384))
    r_x = from_rgb(inputs)
    ########################
    # Fade in block
    ########################
    r_x = downsample_block(r_x, filters1=32, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(640,384))
    r_x = Multiply()([alpha, r_x])
    x = Add()([l_x, r_x])
    ########################
    # Stable blocks
    ########################
    x = downsample_block(x, filters1=64, filters2=128, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(320,192))
    x = downsample_block(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(160,96))
    x = downsample_block(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(80,48))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(40,24))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(20,12))
    x = downsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(10,6))
    x = discriminator_block(x, min_size=3)
    model = Model(inputs=[inputs, alpha], outputs=x)
    return model