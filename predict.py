import tensorflow as tf
slim = tf.contrib.slim
import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from os import listdir
from os.path import isfile, join
from os import walk
os.environ['CUDA_VISIBLE_DEVICES'] = '' #Uncomment this line to run prediction on CPU.

#Inception V1 uses images of 224x224x3
image_size = 224
session = tf.Session()

def get_test_images(mypath):
	return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find('.jpg') != -1]

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]


def predict(train_dir, test_dir, output_file, num_classes):
	""" 
	Loads weights from a TF model and makes predictions.

	Arguments:

	train_dir: directory of trained model

	test_dir: directory of test images (split into folders by class)

	output_file: output file to store predictions

	num_classes: number of classes of prediction

	Returns:

	Outpiuts a file output_file with predictions  
	"""

	train_dir = train_dir
	test_path = test_dir
	output = output_file
	nb_classes = num_classes

	print('Start to read images!')
	image_list = get_test_images(test_path)
	processed_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

	with slim.arg_scope(inception.inception_v1_arg_scope()):
		logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)

	def predict_fn(images):
	    return session.run(probabilities, feed_dict={processed_images: images})

	probabilities = tf.nn.softmax(logits)

	#Loads in latest training checkpoint
	checkpoint_path = tf.train.latest_checkpoint(train_dir)
	init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,slim.get_variables_to_restore())
	init_fn(session)
	print('Start to transform images!')
	images = transform_img_fn(image_list)

	fto = open(output, 'w')
	print('Start doing predictions!')
	preds = predict_fn(images)
	print (len(preds))
	for p in range(len(preds)):
		print (image_list[p], preds[p,:], np.argmax(preds[p,:]))
		fto.write(image_list[p])
		for j in range(len(preds[p,:])):
			fto.write('\t' + str(preds[p, j]))
		fto.write('\n')

	fto.close()

if __name__ == '__main__':
  """ 
  Converts the images in the train folder to TFrecords 
  """
  train_dir = 'TrainedModel/'
  test_dir = 'Images/test'
  output_file = 'Results/output.txt'
  num_classes = 2
  predict(train_dir, test_dir, output_file, num_classes)
