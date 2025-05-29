from turtle import color
import tensorflow as tf
import numpy as np
from config import *
from preprocessing import *
from util import EqualizeLearningRate
from util import *
from generator import *
from discriminator import *
from flask import Flask
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors
import sys
import imageio
from interpolate import generate_image
from train import model_builder
#
# Note:
# convert .gif to .mp4: ffmpeg -i movie.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video.mp4
#
def get_color_vectors(generator, num_examples=10):
    step_counter = 0
    white_count, blue_count, red_count, green_count = 0,0,0,0
    green_vector, blue_vector, red_vector, white_vector = np.zeros([1,NOISE_DIM]), np.zeros([1,NOISE_DIM]), np.zeros([1,NOISE_DIM]), np.zeros([1,NOISE_DIM])
    while(white_count < num_examples or blue_count < num_examples or green_count < num_examples or red_count < num_examples):
        # generate random image and identify color
        tf.keras.backend.clear_session()
        sample_noise = tf.random.normal([1, NOISE_DIM])

        sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)
        latent_input = [sample_noise, sample_alpha]
        image = generator.predict(latent_input)[0]
        color = get_color(image)
        step_counter += 1
        if 1:
            image_height, image_width = image.shape[0], image.shape[1]
            image = image.reshape((image_height,image_width,3))* 0.5 + 0.5
            fig, ax = plt.subplots(1,1)
            ax.imshow(image)
            ax.title.set_text(color)
            plt.show()

        if color == "green" and green_count < num_examples:
            green_count += 1
            green_vector += sample_noise
        elif color =="red" and red_count < num_examples:
            red_count += 1
            red_vector += sample_noise
        elif color == "blue" and blue_count < num_examples:
            blue_count += 1
            blue_vector += sample_noise
        elif color == "white" and white_count < num_examples:
            white_count += 1
            white_vector += sample_noise
        if step_counter % 5 == 0:
            print("red=",red_count,"; blue=", blue_count,"; white=", white_count, "; green=",green_count,"\n")


    
    green_vector /= num_examples
    blue_vector /= num_examples 
    red_vector /= num_examples
    white_vector /= num_examples
    print("steps needed: ", step_counter)
    return green_vector, blue_vector, red_vector, white_vector

def change_color(latent_vector, current_color, target_color, vector_path, res):
    if target_color == current_color:
        return latent_vector
    current_color_vector = np.loadtxt(vector_path+'/'+current_color+'_vector_'+res+'.csv') #.reshape([1, NOISE_DIM])
    target_color_vector = np.loadtxt(vector_path+'/'+target_color+'_vector_'+res+'.csv')   #.reshape([1, NOISE_DIM])

    current_color_vector = tf.convert_to_tensor(current_color_vector, dtype="float32")
    target_color_vector = tf.convert_to_tensor(target_color_vector, dtype="float32")

    current_color_vector = tf.reshape(current_color_vector, [1, NOISE_DIM])
    target_color_vector = tf.reshape(target_color_vector, [1, NOISE_DIM])

    return latent_vector - current_color_vector + target_color_vector

def get_color_from_latent_vector(latent_vector, res_string):
    white_vector = np.linalg.norm( latent_vector - np.loadtxt('white_vector_'+res_string+'.csv').reshape([1, NOISE_DIM]))
    green_vector = np.linalg.norm(latent_vector - np.loadtxt('green_vector_'+res_string+'.csv').reshape([1, NOISE_DIM]))
    blue_vector = np.linalg.norm(latent_vector - np.loadtxt('blue_vector_'+res_string+'.csv').reshape([1, NOISE_DIM]))
    red_vector = np.linalg.norm(latent_vector - np.loadtxt('red_vector_'+res_string+'.csv').reshape([1, NOISE_DIM]))
    color_min = np.argmin([white_vector, green_vector, blue_vector, red_vector ])
    print(color_min)
    color_list = ["white", "green", "blue", "red"]
    return color_list[color_min]

def get_color(image, bins=20):
    image_height, image_width = image.shape[0], image.shape[1]
    num_pixel = (image_height*image_width)
    
    image_np = image.reshape((image_height,image_width,3))* 0.5 + 0.5
    image_np = matplotlib.colors.rgb_to_hsv(image_np)
    image_sat = image_np[:,:,1].reshape(num_pixel)
    image_np = image_np[:,:,0].reshape(num_pixel)
    image_hist = np.histogram(image_np, bins, range=[0., 1.])
    upper_thr, lower_thr = int(0.65*bins),int(0.5*bins)
    image_hist[0][lower_thr:upper_thr] = 0
    image_hist_sat = np.histogram(image_sat, bins, range=[0.,1.])    
    max_color_arg, max_color_val = np.argmax(image_hist[0])*1./bins, np.max(image_hist[0])
    low_sat_portion = np.sum(image_hist_sat[0][:5])/num_pixel
    color = "unknown"
    
    color_identified = max_color_val/num_pixel > 0.05
    white_identified = low_sat_portion > 0.08

    #search for second color:
    image_hist[0][int(bins*max_color_arg)]=0
    second_max_color_arg, second_max_color_val = np.argmax(image_hist[0])*1./bins,np.max(image_hist[0])
    arg_diff = 0.5 - abs(abs(second_max_color_arg - max_color_arg) - 0.5)
    
    second_color_identified = second_max_color_val/num_pixel > 0.05 and arg_diff > 0.2
    if color_identified and not white_identified and not second_color_identified:
        if 0.2 < max_color_arg < 0.43:
            color = "green"
        elif 0.6 < max_color_arg < 0.8:
            color = "blue"
        elif max_color_arg < 0.14 or max_color_arg > 0.8:
            color = "red"
    if white_identified and not color_identified and not second_color_identified:
        color = "white"
    if second_color_identified:
        color = "unknown"
    
    return color
    
def correct_vectors(g,b,r,w):
    indices = np.arange(1, 513, 1).reshape([1, NOISE_DIM])
    print(b.shape, indices.shape)
    plt.clf()
    mean_vector = np.mean(np.array([g,b,r,w]), axis=0)
    g -= mean_vector
    b -= mean_vector
    r -= mean_vector
    w -= mean_vector
    print(mean_vector.shape)
    plt.scatter(indices,b.T,  color='blue')
    plt.scatter(indices, g.T, color='green')
    plt.scatter(indices, r.T, color='red')
    plt.scatter(indices, w.T, color='black')
    plt.show()
def calc_and_save_color_vectors(generator, num_examples=10, path=''):
    sample_noise = tf.random.normal([1, NOISE_DIM])

    sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)
    latent_input = [sample_noise, sample_alpha]
    image = generator.predict(latent_input)[0]
    image_height, image_width = image.shape[0], image.shape[1]
    green_vector, blue_vector, red_vector, white_vector = get_color_vectors(generator, num_examples)
    np.savetxt(path+'/'+'white_vector_'+str(image_height)+'x'+str(image_width)+'.csv', white_vector)
    np.savetxt(path+'/'+'green_vector_'+str(image_height)+'x'+str(image_width)+'.csv', green_vector)
    np.savetxt(path+'/'+'red_vector_'+str(image_height)+'x'+str(image_width)+'.csv', red_vector)
    np.savetxt(path+'/'+'blue_vector_'+str(image_height)+'x'+str(image_width)+'.csv', blue_vector)

def get_latent_vector_from_image(image, model_paths):
    height, width = image.shape[0], image.shape[1]
    generator, _ = model_builder(height)
    generator.load_weights(model_paths+"/"+str(height)+"x"+str(height)+"_generator.h5")
    print(generator._get_trainable_state())
    for layer in generator.layers:
        print(layer.name, layer.trainable)

if __name__ == "__main__":
    plt.clf()
    generator = build_80x48_generator()
    generator.load_weights("/home/manuel/progan/11_07_2022_models_4_color/48x48_generator.h5")
    #image = get_4_color_im('2021-03-08-10-49-57-942781_3.png',24,40)
    #get_latent_vector_from_image(image[0], 'models_4_color')
    calc_and_save_color_vectors(generator, 100, 'vectors_final')

    white_vector = np.loadtxt('white_vector_24x40.csv').reshape([1, NOISE_DIM])
    green_vector = np.loadtxt('green_vector_24x40.csv').reshape([1, NOISE_DIM])
    blue_vector = np.loadtxt('blue_vector_24x40.csv').reshape([1, NOISE_DIM])
    red_vector = np.loadtxt('red_vector_24x40.csv').reshape([1, NOISE_DIM])
    correct_vectors(green_vector, blue_vector, red_vector, white_vector)
    
    sample_noise1 = tf.random.normal([1, NOISE_DIM])

    green_to_red_noise = sample_noise1 - red_vector + green_vector
    sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)
    latent_input = [sample_noise1, sample_alpha]
    image = generator.predict(latent_input)[0]
    image = image.reshape((24,40,3))* 0.5 + 0.5

    plt.imshow(image)
    plt.title("color: "+get_color_from_latent_vector(sample_noise1, 24, 40))
    plt.show()
    latent_input = [green_to_red_noise, sample_alpha]
    image = generator.predict(latent_input)[0]
    image = image.reshape((24,40,3))* 0.5 + 0.5
    plt.imshow(image)
    plt.show()

    images = []
    images.append(generate_image(sample_noise1, generator))

    for t in list(np.linspace(0, 1, 10)):
        sample_noise = sample_noise1 + t*(green_to_red_noise - sample_noise1)
        images.append(generate_image(sample_noise, generator))

    images.append(generate_image(green_to_red_noise, generator))

    imageio.mimsave('red_to_green' + '.gif', images, duration=0.5)
