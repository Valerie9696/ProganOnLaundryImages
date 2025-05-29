from config import *
from preprocessing import *
from util import EqualizeLearningRate
from util import *
from generator import *
from discriminator import *

def model_builder(target_resolution):
    '''
        Helper function to build models
    '''
    generator = None
    discriminator = None
    if target_resolution == 4:
        generator = build_4x4_generator()
        discriminator = build_4x4_discriminator()
    if target_resolution == 3:
        generator = build_5x3_generator()
        discriminator = build_5x3_discriminator()
    elif target_resolution == 8:
        generator = build_8x8_generator()
        discriminator = build_8x8_discriminator()
    elif target_resolution == 6:
        generator = build_10x6_generator()
        discriminator = build_10x6_discriminator()
    elif target_resolution == 16:
        generator = build_16x16_generator()
        discriminator = build_16x16_discriminator()
    elif target_resolution == 12:
        generator = build_20x12_generator()
        discriminator = build_20x12_discriminator()
    elif target_resolution == 32:
        generator = build_32x32_generator()
        discriminator = build_32x32_discriminator()
    elif target_resolution == 24:
        generator = build_40x24_generator()
        discriminator = build_40x24_discriminator()
    elif target_resolution == 64:
        generator = build_64x64_generator()
        discriminator = build_64x64_discriminator()
    elif target_resolution == 48:
        generator = build_80x48_generator()
        discriminator = build_80x48_discriminator()
    elif target_resolution == 128:
        generator = build_128x128_generator()
        discriminator = build_128x128_discriminator()
    elif target_resolution == 96:
        generator = build_160x96_generator()
        discriminator = build_160x96_discriminator()
    elif target_resolution == 256:
        generator = build_256x256_generator()
        discriminator = build_256x256_discriminator()
    elif target_resolution == 192:
        generator = build_320x192_generator()
        discriminator = build_320x192_discriminator()
    elif target_resolution == 512:
        generator = build_512x512_generator()
        discriminator = build_512x512_discriminator()
    elif target_resolution == 384:
        generator = build_640x384_generator()
        discriminator = build_640x384_discriminator()
    else:
        print("target resolution models are not defined yet")
    return generator, discriminator

generator, discriminator = model_builder(image_size)
generator.summary()
#plot_model(generator, show_shapes=True, dpi=64)
discriminator.summary()
#plot_model(discriminator, show_shapes=True, dpi=64)

D_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
G_optimizer = Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, D_optimizer, G_optimizer):
    '''
        Set new learning rate to optimizers
    '''
    K.set_value(D_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)
    
def calculate_batch_size(image_size):
    if image_size < 64:
        return 16
    elif image_size < 128:
        return 12
    elif image_size == 128:
        return 8
    elif image_size == 256:
        return 4
    else:
        return 3

def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    # Test input is a list include noise and label
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, '{}x{}_image_at_epoch_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))
    #plt.show()

num_examples_to_generate = 9

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM], seed=0)
sample_alpha = np.repeat(1, num_examples_to_generate).reshape(num_examples_to_generate, 1).astype(np.float32)
generate_and_save_images(generator, 0, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=False, is_flatten=False)

@tf.function
def WGAN_GP_train_d_step(generator, discriminator, real_image, alpha, batch_size, step):
    '''
        One training step
        
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator([noise, alpha], training=True)
            #print("real_image: ", real_image.shape, "| fake image: ", fake_image.shape)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator([fake_image_mixed, alpha], training=True)
            
        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
        
        fake_pred = discriminator([fake_image, alpha], training=True)
        real_pred = discriminator([real_image, alpha], training=True)
        
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,
                                            discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    D_optimizer.apply_gradients(zip(D_gradients,
                                                discriminator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)
            
@tf.function
def WGAN_GP_train_g_step(generator, discriminator, alpha, batch_size, step):
    '''
        One training step
        
        Reference: https://www.tensorflow.org/tutorials/generative/dcgan
    '''
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator([noise, alpha], training=True)
        fake_pred = discriminator([fake_image, alpha], training=True)
        G_loss = -tf.reduce_mean(fake_pred)
    # Calculate the gradients for discriminator
    G_gradients = g_tape.gradient(G_loss,
                                            generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                                generator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('G_loss', G_loss, step=step)

