import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization

from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorboard_utils import CustomModelSaver
from tensorflow_addons.layers import SpatialPyramidPooling2D as SPP

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_19

from tensorflow.nn import softmax as Softmax

import hyperparameters as hp
from os.path import abspath

# from region_proposal import RegionProposalNetwork as RPN
from roi_pooling import ROI_Pooling
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import register_keras_serializable

import pdb #for debugging



def learning_rate_scheduler(epoch):

    if epoch<hp.lr_threshold:
        return hp.learning_rate_1

    else:
        return hp.learning_rate_2

def bounding_box_iou(boxes1, boxes2):
    ''' Calculates IoU for pairs of bounding boxes from region proposals '''

    boxes1 = tf.cast(boxes1, dtype=tf.float32)
    boxes2 = tf.cast(boxes2, dtype=tf.float32)

    b1_x1 = boxes1[:,0]; b1_x2 = boxes1[:,1]; b1_y1 = boxes1[:,2]; b1_y2 = boxes1[:,3]

    b2_x1 = boxes2[:,0]; b2_x2 = boxes2[:,1]; b2_y1 = boxes2[:,2]; b2_y2 = boxes2[:,3]


    intersect_x1 = tf.reduce_max(tf.stack([b1_x1, b2_x1], axis=1), axis=1)
    intersect_y1 = tf.reduce_max(tf.stack([b1_y1, b2_y1], axis=1), axis=1)

    intersect_x2 = tf.reduce_min(tf.stack([b1_x2, b2_x2],axis=1), axis=1)
    intersect_y2 = tf.reduce_min(tf.stack([b1_y2, b2_y2],axis=1), axis=1)

    #intersection area
    intersect_area = tf.clip_by_value(intersect_x2-intersect_x1+1, 0.0, float("inf"))*tf.clip_by_value(intersect_y2-intersect_y1+1, 0.0, float("inf"))

    #area of bounding boxes separately
    area_boxes1 = tf.clip_by_value(b1_x2-b1_x1+1, 0.0, float("inf"))*tf.clip_by_value(b1_y2-b1_y1+1, 0.0, float("inf"))

    area_boxes2 = tf.clip_by_value(b2_x2-b2_x1+1, 0.0, float("inf"))*tf.clip_by_value(b2_y2-b2_y1+1, 0.0, float("inf"))

    #union area
    union_area = area_boxes1 + area_boxes2 - intersect_area

    iou = tf.cast(intersect_area, dtype = tf.float32)/tf.cast(union_area, dtype=tf.float32)

    return iou


class WeaklySupervisedDetection(tf.keras.Model):



    def __init__(self, backbone):
        super(WeaklySupervisedDetection, self).__init__()

        #create learning rate scheduler
        values = [hp.learning_rate_1, hp.learning_rate_2, hp.learning_rate_2]
        boundaries = [hp.num_epochs//2, hp.num_epochs]
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

        #optimizer to use when optimizing the energy function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)


        #setup backbone architecture: pre-trained VGG network
        if backbone=='VGG16':
            self.backbone = VGG16(weights = 'imagenet', include_top=False, pooling=None, input_shape = (200,200,3))
            self.preprocess = preprocess_input_16
            self.feat_map_scaling = 16

        elif backbone=='VGG19':
            self.backbone = VGG19(weights = 'imagenet', include_top=False, pooling = None, input_shape = (200,200,3))
            self.preprocess = preprocess_input_19
            self.feat_map_scaling = 19

        #set layers as non trainable: only leave last 2 layers to be trainable
        for layer in self.backbone.layers[:-3]:
            layer.trainable = False



        #main WSDDN model, post VGG16 backbone
        self.wsddn_layers = [

            #ROI pooling layer
            ROI_Pooling,


            #batch normalization
            BatchNormalization(),

            #flatten layer
            Flatten(),
        ]

        #intermediate dense layers after extracting region-wise features (pooling layer)
        self.fc6 = Dense(9216, activation='relu', name="block6_fc")
        self.fc7 = Dense(4096, activation = 'relu', name="block7_fc")
        self.fc7_5 = Dense(2048, activation='relu', name="block7.5_fc")


        #classification branch layers: performed on individual regions
        #linear map to C for each region |R|: Cx|R| mapping across all regions
        self.fc8c = Dense(hp.num_classes, name='block8_fcc')


        #detection branch layers
        self.fc8d = Dense(hp.num_classes, name='block8_fcd')


    def rpn_layer(self, base_layers):
        """Create a rpn layer
            Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                    Keep the padding 'same' to preserve the feature map's size
            Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                    classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                    regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
        Args:
            base_layers: vgg in here
            num_anchors: 9 in here

        Returns:
            [x_class, x_regr, base_layers]
            x_class: classification for whether it's an object
            x_regr: bboxes regression
            base_layers: vgg in here
        """
        x = Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

        x_class = Conv2D(hp.num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
        x_regr = Conv2D(hp.num_anchors * 4, (1, 1), activation='sigmoid', kernel_initializer='glorot_uniform', name='rpn_out_regress')(x)

        return [x_class, x_regr, base_layers]



    def call(self, image, labels, spatial_reg=False):
        '''Performs the forward pass of the WSDDN on input 'x' '''
        pdb.set_trace()

        '''VGG backbone and pooling: pre SPP or ROI pool'''
        #running input on the base pre-trained VGG16 or VGG19 model
        image = self.preprocess(image)
        backbone_pre_pooling_output = self.backbone(image)

        # block5_pooling_output = MaxPool2D(2, name="block5_pool")(backbone_pre_pooling_output)
        '''Get the regions of interest (ROIs) using the RPN layer '''
        [objectness, original_rois, backbone_pre_pooling_output] = self.rpn_layer(backbone_pre_pooling_output)


        original_rois = tf.reshape(tf.identity(original_rois), (1, backbone_pre_pooling_output.shape[1], backbone_pre_pooling_output.shape[2], -1, 4))
        original_rois = original_rois[(objectness > hp.objectness_threshold)]
        #indexes the original_rois using the valid_objectness ROIs



        '''Resizing the ROIs and pooling original proposals'''
        backbone_pre_pooling_output = tf.squeeze(backbone_pre_pooling_output)

        x1 = original_rois[:,0]*(backbone_pre_pooling_output.shape[0])
        x2 = original_rois[:,1]*(backbone_pre_pooling_output.shape[0])
        y1 = original_rois[:,2]*(backbone_pre_pooling_output.shape[1])
        y2 = original_rois[:,3]*(backbone_pre_pooling_output.shape[1])

        original_rois = tf.stack([x1,x2,y1,y2], axis=1)
        #use absolute value to convert + remove negative region proposals
        #Note: original_rois shape (num rois, 4)

        rois = []
        filtered_origin_rois = []


        for idx,roi in enumerate(original_rois):

            #extract ROI coordinates
            (x1, x2, y1, y2) = tf.cast(roi, dtype=tf.int32)



            #roi in the form (x1,x2,y1,y2)
            if x2-x1<=0 or y2-y1<=0:
                continue


            #filter out the ROI region from the feature map output
            height = y2-y1; width = x2-x1; channels = backbone_pre_pooling_output.shape[2]

            roi_feature = tf.slice(backbone_pre_pooling_output, begin = tf.stack([y1,x1,0]), size = tf.stack([height, width, channels]) )

            '''replace with spatial pyramidal pooling (SPP) in wsddn_layers'''
            roi_feature = tf.image.resize(roi_feature, (hp.roi_pooling_output[0], hp.roi_pooling_output[1]))

            #convert the standard size ROIs to a flat vector of values, accumulate and pass into dense layers
            roi_feature = tf.reshape(roi_feature, -1)

            #extracting coordinates of the pooled ROIs
            if not rois:
                rois_feature = tf.expand_dims(roi_feature, axis=0)

            else:
                rois_feature = tf.concat( (rois_feature, tf.expand_dims(roi_feature,axis=0)), axis=0 )


            # accumulate a list of ROI coordinates
            rois.append( tf.cast(roi, dtype=tf.int32) )
            filtered_origin_rois.append(tf.cast(original_rois[idx], dtype=tf.int32).numpy())

        pdb.set_trace()
        #if no rois collected or if rois is empty list
        if not rois:
            return (None, None, None, None)


        filtered_origin_rois = tf.convert_to_tensor(filtered_origin_rois)

        '''Linear FCN after region proposal Pooling'''

        #after getting features from pooling, run the fully connected layers + perform softmax
        fc_output = self.fc6(rois_feature)
        fc_output = self.fc7(fc_output)
        fc_output = self.fc7_5(fc_output)

        #classification branch + softmax: output num_regions * num classes
        fc_class_out = self.fc8c(fc_output)
        fc_class_out = Softmax(fc_class_out, axis=0)


        #detection branch + softmax: output num regions * num classes
        fc_detect_out = self.fc8d(fc_output)
        fc_detect_out = Softmax(fc_detect_out, axis=1)


        #hadamard product (elementwise) across detection and classification outputs
        scores = tf.multiply(fc_class_out, fc_detect_out)
        output = tf.reduce_sum(scores, axis=0) #summing scores for each class: across the regions

        #clipping outputs within 0-1 range
        output = tf.clip_by_value(output, 0.0, 1.0, name="clipping_scores")


        #add spatial regularization if needed
        spatial_regularizer_output = 0

        pdb.set_trace()

        if spatial_reg:
            spatial_regularizer_output = self.spatial_regularizer(scores, labels, fc_output, tf.convert_to_tensor(rois))

        return output, scores, filtered_origin_rois, spatial_regularizer_output






    @staticmethod
    def spatial_regularizer(scores, labels, fc7, rois):
        '''Performs spatial regularization on the final scores + adds to the energy function optimization '''

        labels = tf.squeeze(labels)

        num_regions = rois.shape[0]

        regularizer_sum = 0

        for k in range(hp.num_classes):

            #find the positive samples for each class: this does not apply for ATARI games as all imgaes have objects with labels
            if labels[k]==0:
                continue


            #sort scores for the class 'k' across regions
            #note: scores of shape (num regions, num classes)
            sorted_regions = tf.argsort(scores[:,k])


            #access region and fc7 layer output with highest score
            highest_score_region = rois[sorted_regions[0]]
            highest_score_fc7_output = fc7[sorted_regions[0]]

            #access other regions and fc7 layer outputs (with non highest scores)
            other_score_region = tf.gather(rois,sorted_regions[1:])
            other_score_fc7_output = tf.gather(fc7, sorted_regions[1:])

            #creating filter for bounding box regions with low IoU score
            region_ious = bounding_box_iou(other_score_region, tf.tile(tf.expand_dims(highest_score_region, axis=0),[num_regions-1,1]) )
            region_ious = tf.cast(region_ious>hp.spatial_reg_iou_threshold, dtype=tf.float32)
            pdb.set_trace()
            #mask out fc7 output for regions with < threshold IoU
            fc7_output_mask = tf.tile(tf.expand_dims(region_ious,axis=1), (1,fc7.shape[1]))
            other_score_fc7_output = other_score_fc7_output*fc7_output_mask




            difference = other_score_fc7_output - highest_score_fc7_output

            difference = difference*tf.expand_dims(tf.gather(scores[:,k], sorted_regions[1:]),  axis=1)

            regularizer_sum += tf.reduce_sum(tf.pow(difference, 2))*0.5


        return hp.spatial_reg_weight*(regularizer_sum/hp.num_classes)




    def l2_regularizer(self):
        '''Performs variation of L2 regularization i.e. (lambda/2) * ||w|| ^2'''


        weight_squared_sum = 0

        all_layers = self.backbone.layers + [self.fc6, self.fc7, self.fc7_5, self.fc8c, self.fc8d]

        #sum of all weights across all layers on the WSDDN model
        for layer in all_layers:

            weight_squared_sum = tf.reduce_sum(tf.math.square(layer.get_weights()))


        return (hp.weight_decay/2)*weight_squared_sum



    def energy_fn(probabilities):
        '''Energy function to optimize by the model for a single iamge'''

        #since all the classes will be present in the screen, the loss reduces to log prob. sum
        return tf.sum(tf.log(probabilities)) + self.l2_regularizer()


    @staticmethod
    def crossentropy_loss(probabilities, label):
        '''Loss function to optimize by the model for a single image'''

        bce = BinaryCrossentropy()

        return bce(probabilities, label)
