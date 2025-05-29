from config import *
from preprocessing import *
from util import *

def generator_input_block(x):
    '''
        Generator input block
    '''
    x = EqualizeLearningRate(Dense(3*5*512, kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_dense')(x)
    x = PixelNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((3, 5, 512))(x)
    x = EqualizeLearningRate(Conv2D(512, 3, strides=1, padding='same',
                                          kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='g_input_conv2d')(x)
    x = PixelNormalization()(x)
    x = LeakyReLU()(x)
    return x

def build_4x4_generator(noise_dim=NOISE_DIM):
    '''
        4 * 4 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    # Not used in 4 * 4, put it here in order to keep the input here same as the other models
    alpha = Input((1), name='input_alpha')
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    
    rgb_out = to_rgb(x)
    model = Model(inputs=[inputs, alpha], outputs=rgb_out)
    return model


def build_5x3_generator(noise_dim=NOISE_DIM):
    '''
        5 * 3 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    # Not used in 4 * 4, put it here in order to keep the input here same as the other models
    alpha = Input((1), name='input_alpha')
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(5, 3))
    
    rgb_out = to_rgb(x)
    model = Model(inputs=[inputs, alpha], outputs=rgb_out)
    return model

def build_8x8_generator(noise_dim=NOISE_DIM):
    '''
        8 * 8 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_10x6_generator(noise_dim=NOISE_DIM):
    '''
        10 * 6 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(5, 3))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(10, 6))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_16x16_generator(noise_dim=NOISE_DIM):
    '''
        16 * 16 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_20x12_generator(noise_dim=NOISE_DIM):
    '''
        20 * 12 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(10, 6))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(20, 12))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_32x32_generator(noise_dim=NOISE_DIM):
    '''
        32 * 32 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(16, 16))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_40x24_generator(noise_dim=NOISE_DIM):
    '''
        40 * 24 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(40, 24))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(20, 12))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(40, 24))

    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_64x64_generator(noise_dim=NOISE_DIM):
    '''
        64 * 64 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(32, 32))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_80x48_generator(noise_dim=NOISE_DIM):
    '''
        80 * 48 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(40, 24))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(80, 48))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(40, 24))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(80, 48))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_128x128_generator(noise_dim=NOISE_DIM):
    '''
        128 * 128 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(64, 64))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_160x96_generator(noise_dim=NOISE_DIM):
    '''
        160 * 96 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(40, 24))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(80, 48))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(160, 96))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(80, 48))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(160, 96))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_256x256_generator(noise_dim=NOISE_DIM):
    '''
        256 * 256 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=128, filters2=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(128, 128))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_320x192_generator(noise_dim=NOISE_DIM):
    '''
        320 * 192 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(40, 24))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(80, 48))
    x, _ = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(160, 96))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=128, filters2=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(320, 192))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(160, 96))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(320, 192))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_512x512_generator(noise_dim=NOISE_DIM):
    '''
        512 * 512 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(16, 16))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(32, 32))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(64, 64))
    x, _ = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(128, 128))
    x, _ = upsample_block(x, filters1=128, filters2=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(256, 256))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=64, filters2=32, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(512, 512))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(256, 256))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(512, 512))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model

def build_640x384_generator(noise_dim=NOISE_DIM):
    '''
        640 * 384 Generator
    '''
    # Initial block
    inputs = Input(noise_dim)
    x = generator_input_block(inputs)
    alpha = Input((1), name='input_alpha')
    ########################
    # Stable blocks
    ########################
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(10, 6))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(20, 12))
    x, _ = upsample_block(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(40, 24))
    x, _ = upsample_block(x, filters1=512, filters2=256, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(80, 48))
    x, _ = upsample_block(x, filters1=256, filters2=128, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(160, 96))
    x, _ = upsample_block(x, filters1=128, filters2=64, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(320, 192))
    ########################
    # Fade in block
    ########################
    x, up_x = upsample_block(x, filters1=64, filters2=32, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(640, 384))
    
    previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(320, 192))
    to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(640, 384))
    
    l_x = previous_to_rgb(up_x)
    r_x = to_rgb(x)
    ########################
    # Left branch in the paper
    ########################
    l_x = Multiply()([1 - alpha, l_x])
    ########################
    # Right branch in the paper
    ########################
    r_x = Multiply()([alpha, r_x])
    combined = Add()([l_x, r_x])
    
    model = Model(inputs=[inputs, alpha], outputs=combined)
    return model
