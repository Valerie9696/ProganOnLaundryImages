#!/usr/bin/env python3

from config import *
from preprocessing import *
from util import EqualizeLearningRate
from util import *
from generator import *
from discriminator import *
from classify_latentspace_vectors import *
from flask import Flask
import io
import random
from flask import Response
from flask import request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)

@app.route("/image")
def get_image():
	res = request.args.get('res')
	if request.args.get('seed'):
		seed = int(request.args.get('seed'))
	else:
		seed = None
	fig = generate(res, seed)
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')

@app.route("/image_color")
def interpolate_color():
	res = request.args.get('res')
	target_color = request.args.get('target_color')
	if request.args.get('seed'):
		seed = int(request.args.get('seed'))
	else:
		seed = None
	latent_vector, sample_noise = get_latent_vector(seed)
	generator = load_generator(res)
	prediction = generator.predict(latent_vector)[0]

	current_color = get_color_from_latent_vector(sample_noise, res)
	latent_vector = change_color(latent_vector, current_color, target_color,'.', res)
	fig = generate_from_lsvector(res, latent_vector)
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')

@app.route("/interpolate")
def interpolate():
	res = request.args.get('res')
	seed1 = int(request.args.get('seed1'))
	seed2 = int(request.args.get('seed2'))
	t = float(request.args.get('t'))

	generator = load_generator(res)

	tf.keras.backend.clear_session()
	sample_noise1 = tf.random.normal([1, NOISE_DIM], seed=seed1)

	tf.keras.backend.clear_session()
	sample_noise2 = tf.random.normal([1, NOISE_DIM], seed=seed2)

	sample_noise = sample_noise1 + t*(sample_noise2 - sample_noise1)
	sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)

	test_input = [sample_noise, sample_alpha]
	prediction = generator.predict(test_input)

	fig = plt.figure(frameon=False, facecolor='white')
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	ax.imshow(prediction[0] * 0.5 +0.5)

	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	return Response(output.getvalue(), mimetype='image/png')


def load_generator(res):
	if res == "3x5":
		generator = build_5x3_generator()
		generator.load_weights("11_07_2022_models_4_color/3x3_generator.h5")
	elif res == "6x10":
		generator = build_10x6_generator()
		generator.load_weights("11_07_2022_models_4_color/6x6_generator.h5")
	elif res == "12x20":
		generator = build_20x12_generator()
		generator.load_weights("11_07_2022_models_4_color/12x12_generator.h5")
	elif res == "24x40":
		generator = build_40x24_generator()
		generator.load_weights("11_07_2022_models_4_color/24x24_generator.h5")
	elif res == "48x80":
		generator = build_80x48_generator()
		generator.load_weights("11_07_2022_models_4_color/48x48_generator.h5")
	elif res == "96x160":
		generator = build_160x96_generator()
		generator.load_weights("11_07_2022_models_4_color/96x160_generator.h5")
	elif res == "192x320":
		generator = build_320x192_generator()
		generator.load_weigths("11_07_2022_models_4_color/192x320_generator.h5")
	elif res == "384x640":
		generator = build_640x384_generator()
		generator.load_weights("11_07_2022_models_4_color/384x640_generator.h5")
	else:
		generator = None
		abort(404)

	return generator

def generate_from_lsvector(res, lsv):

	generator = load_generator(res)

	sample_noise = lsv
	sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)

	test_input = [sample_noise, sample_alpha]
	prediction = generator.predict(test_input)

	fig = plt.figure(frameon=False, facecolor='white')
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	ax.imshow(prediction[0] * 0.5 +0.5)

	return fig

def generate(res, seed):

	generator = load_generator(res)

	tf.keras.backend.clear_session()
	sample_noise = tf.random.normal([1, NOISE_DIM], seed=seed)
	sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)

	test_input = [sample_noise, sample_alpha]
	prediction = generator.predict(test_input)

	fig = plt.figure(frameon=False, facecolor='white')
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	ax.imshow(prediction[0] * 0.5 +0.5)

	return fig

def get_latent_vector(seed):
	tf.keras.backend.clear_session()
	sample_noise = tf.random.normal([1, NOISE_DIM], seed=seed)
	sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)
	latent_space = [sample_noise, sample_alpha]
	return latent_space, sample_noise

