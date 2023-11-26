import numpy as np

from keras.models import load_model
from keras import backend

def remove_stars(b, path_to_model):
	'''
	Function that performs star removal for nighttime data using a neural network.
	This function operates on 3d (stripe, epoch, altitude) orbit images
	individually.

	Args:
		b (ndarray)				: (stripe, epoch, altitude) array of brightness profiles in
									Rayleighs for a particular orbit.
									Hot and cold pixel correctiond have to be applied to the profiles
									before to call this function for a good performance.
		path_to_model (string)	: path to the neural network model trained to remove stars from b.
								The path can be a hdf5 file or a directory.

	Returns:
		b_corrected (ndarray): star removed profiles
	'''

	# #Upload the model only the first time
	# if __remove_stars__.model is None:
	# 	model = load_model(path_cnn)
	# 	__remove_stars__.model = model
	# else:
	# 	model = __remove_stars__.model

	model = load_model(path_to_model)
	# swap epoch and altitude axes to match model input format
	b = np.transpose(b, (1,2,0))

	# keras model needs an additional dimension channel
	# (epoch, altitude, stripe, color)
	b_corrected = model.predict(b[:,:,:,None])

	#Get rid of the extra dimension channel
	b_corrected = b_corrected[:,:,:,0]

	#Return dimension to the original format
	b_corrected = np.transpose(b_corrected, (2,0,1) )

	return(b_corrected)

# __remove_stars__.model = None
