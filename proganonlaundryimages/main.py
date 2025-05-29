#!/usr/bin/env python3

from config import *
from preprocessing import *
from util import *
from generator import *
from discriminator import*
from train import *

train_data = generate_dataset(image_size, image_width, batch_size)
# Load previous resolution model
if image_size > 4:
    if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_width / 2)))):
        generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size / 2), int(image_width / 2))), by_name=True)
        print("generator loaded")
    if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_width / 2)))):
        discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size / 2), int(image_width / 2))), by_name=True)
        print("discriminator loaded")


# To resume training, comment it if not using.
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_width)))):
    generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(int(image_size), int(image_width))), by_name=False)
    print("generator loaded")
if os.path.isfile(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_width)))):
    discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(int(image_size), int(image_width))), by_name=False)
    print("discriminator loaded")

current_learning_rate = LR
training_steps = math.ceil(total_data_number / batch_size)
# Fade in half of switch_res_every_n_epoch epoch, and stablize another half
alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
alpha = min(1., (CURRENT_EPOCH - 1) % switch_res_every_n_epoch * training_steps *  alpha_increment)


for epoch in range(CURRENT_EPOCH, EPOCHs + 1):
    

    start = time.time()
    print('Start of epoch %d' % (epoch,))
    print('Current alpha: %f' % (alpha,))
    print('Current resolution: {} * {}'.format(image_size, image_width))
    # Using learning rate decay
    # current_learning_rate = learning_rate_decay(current_learning_rate)
    # print('current_learning_rate %f' % (current_learning_rate,))
    # set_learning_rate(current_learning_rate) 

    for step, (image) in enumerate(train_data):
        if PROFILE:
            tf.profiler.experimental.start("profil_log", options=PROF_OPTIONS)
        current_batch_size = image.shape[0]
        alpha_tensor = tf.constant(np.repeat(alpha, current_batch_size).reshape(current_batch_size, 1), dtype=tf.float32)
        # Train step

        WGAN_GP_train_d_step(generator, discriminator, image, alpha_tensor,
                             batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
        WGAN_GP_train_g_step(generator, discriminator, alpha_tensor,
                             batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))


        # update alpha
        alpha = min(1., alpha + alpha_increment)

        if step % 10 == 0:
            print ('.', end='')
        if PROFILE:
            tf.profiler.experimental.stop()
    # Clear jupyter notebook cell output
    clear_output(wait=True)
    # Using a consistent image (sample_X) so that the progress of the model is clearly visible.
    generate_and_save_images(generator, epoch, [sample_noise, sample_alpha], figure_size=(6,6), subplot=(3,3), save=True, is_flatten=False)

    if epoch % SAVE_EVERY_N_EPOCH == 0:
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_width)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_width)))
        print ('Saving model for epoch {}'.format(epoch))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                      time.time()-start))
    # Train next resolution
    if epoch % switch_res_every_n_epoch == 0:
        print('saving {} * {} model'.format(image_size, image_width))
        generator.save_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(image_size, image_width)))
        discriminator.save_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(image_size, image_width)))
        # Reset alpha
        alpha = 0
        previous_image_size = int(image_size)
        previous_image_width = int(image_width)
        image_size = int(image_size * 2)
        image_width = int(image_width * 2)
        if image_size > 384:
            print('Resolution reach 384x640, finish training')
            break
        print('creating {} * {} model'.format(image_size, image_width))
        generator, discriminator = model_builder(image_size)
        generator.load_weights(os.path.join(MODEL_PATH, '{}x{}_generator.h5'.format(previous_image_size, previous_image_width)), by_name=True)
        discriminator.load_weights(os.path.join(MODEL_PATH, '{}x{}_discriminator.h5'.format(previous_image_size, previous_image_width)), by_name=True)

        print('Making {} * {} dataset'.format(image_size, image_width))
        batch_size = calculate_batch_size(image_size)
        train_data = generate_dataset(image_size, image_width, batch_size)
        training_steps = math.ceil(total_data_number / batch_size)
        alpha_increment = 1. / (switch_res_every_n_epoch / 2 * training_steps)
        print('start training {} * {} model'.format(image_size, image_size))
    


