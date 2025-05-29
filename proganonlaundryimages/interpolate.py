#!/usr/bin/env python3

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
import imageio
import sys

#
# Note:
# convert .gif to .mp4: ffmpeg -i movie.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video.mp4
#

def generate_image(sample_noise, generator):
	sample_alpha = np.repeat(1, 1).reshape(1, 1).astype(np.float32)
	test_input = [sample_noise, sample_alpha]
	prediction = generator.predict(test_input)
	fig = plt.figure(frameon=False, facecolor='white')
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(prediction[0] * 0.5 + 0.5)
	output = io.BytesIO()
	FigureCanvas(fig).print_png(output)
	image_bytes = imageio.imread(output.getvalue())
	plt.close(fig)
	return image_bytes

if __name__ == "__main__":
	if len(sys.argv) != 6:
		print(f"usage: {sys.argv[0]} <resolution> <seed1> <seed2> <steps> <delay>")
		print(f"example: {sys.argv[0]} 24x40 1004 777 10 0.5")
		sys.exit(1)

	res = sys.argv[1]
	seed1 = int(sys.argv[2])
	seed2 = int(sys.argv[3])
	steps = int(sys.argv[4])
	delay = float(sys.argv[5])

	generator = build_40x24_generator()
	generator.load_weights("models/24x40_generator.h5")

	tf.keras.backend.clear_session()
	sample_noise1 = tf.random.normal([1, NOISE_DIM], seed=seed1)

	tf.keras.backend.clear_session()
	sample_noise2 = tf.random.normal([1, NOISE_DIM], seed=seed2)

	images = []
	images.append(generate_image(sample_noise1, generator))

	for t in list(np.linspace(0, 1, steps)):
		sample_noise = sample_noise1 + t*(sample_noise2 - sample_noise1)
		images.append(generate_image(sample_noise, generator))

	images.append(generate_image(sample_noise2, generator))

	imageio.mimsave(res + '_' + str(seed1) + '_' + str(seed2) + '_' + str(steps) + '_' + str(delay) + '.gif', images, duration=delay)
