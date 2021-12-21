
'''
All hyperparameters to the WSDDN model
'''

#number of epochs to run training
num_epochs = 20

#frequency of batches with which to record validation results
validation_batch_freq = 10

#learning rate for the 1st half of the epochs
learning_rate_1 = 5e-4
#learning rate for the 2nd half of the epochs
learning_rate_2 = 5e-5


momentum = 0.01

#image size to rescale the input images to: 64x64 in DreamerV2
# img_size = 64

#update the weights every batch_size number of elements
#i.e. process batch_size number of images at a time for recognition
batch_size = 20

roi_pool_shape = [6,6]

#number of classes to identify for the particular ATARI game
num_classes = 2

#weight for L2 regularization + weight for spatial regularization
weight_decay = 1.0
spatial_reg_weight = 5e-4


#number of anchor boxes to apply to each anchor point
num_anchors = 9


objectness_threshold = 0.6

roi_pooling_output = [6,6]
reshaped_image_size = (200,200)


#compares the feature map discrepancies at fc7 for regions with IoU > threshold
spatial_reg_iou_threshold = 0.8

#for the sake of separating out graphical outputs + weights
experiment_number = 1

#weights for triplet loss per class
triplet_soft_margin = 1e-5
