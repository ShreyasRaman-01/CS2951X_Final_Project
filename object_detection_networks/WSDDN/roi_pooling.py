import tensorflow as tf
import math
import tensorflow_addons as tfa

import hyperparameters as hp


def ROI_Pooling(feature_map):

    '''Normalizes the input feature map to a fixed output shape'''

    roi_pooling = tfa.layers.AdaptiveMaxPooling2D(output_size = (hp.roi_pooling_output[0], hp.roi_pooling_ouput[1]))

    #apply pooling to the base feature maps from VGG network
    pooling_output = roi_pooling(feature_map)


    return pooling_output
