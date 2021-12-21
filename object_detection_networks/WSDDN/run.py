import os
import sys
import argparse
import re
from datetime import datetime
from PIL import Image
import hyperparameters as hp
from wsddn import WeaklySupervisedDetection as WSDDN_Model, learning_rate_scheduler
from matplotlib import pyplot as plt
import numpy as np

#import for logging metrics + losses
import pandas as pd

#for progress bar with metrics update
from keras.utils import generic_utils
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import time
#for debugging
import pdb


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH_TO_DATA = '/home/shreyas_sundara_raman/CS2951X_Final_Project/object_detection_networks/data'
PATH_TO_WEIGHTS = '/home/shreyas_sundara_raman/CS2951X_Final_Project/object_detection_networks/WSDDN/weights'

#create the weights saving path if it doesn't exist
if not os.path.isdir(os.path.join(PATH_TO_WEIGHTS,str(hp.experiment_number))):
    os.makedirs(os.path.join(PATH_TO_WEIGHTS,str(hp.experiment_number)))

#callbacks to log metrics and save weights at checkpoints
# callback_list = [tf.keras.callbacks.TensorBoard(log_dir=logs_path,update_freq='batch',profile_batch=0),
#                  LearningRateScheduler(learning_rate_scheduler),
#                  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)]
#
# history = model.fit(X, y, epochs=hp.num_epochs, callbacks=[callback_list])



class DatasetCreator:

    def __init__(self, atari_game, image_size):

        #lists storing training and testing data in the format: [ batch of image filepaths, batch of class labels ]
        self.test_data = []
        self.train_data = []


        #select size of labels space vector based on game
        self.num_classes = 0

        if atari_game=='MsPacman-v0':
            self.num_classes = 2
            #pacman and ghost

        elif atari_game=='BreakoutDeterministic-v4':
            self.num_classes = 2
            #paddle and ball

        #save the game name for the dataset path
        self.game = atari_game

        #store the input image render size
        self.size = image_size


    def create_datasets(self, path_to_data):

        '''
        Reads and split a train + testing set from the screenshots of ATARI games

        Args:
            - full path to dataset

        Return: None
            - adds list containing test_data and train_data i.e. list of image filepaths and class label vectors
            to the self.train_data and self.test_data variables
        '''

        #setting up the class label vector
        class_label = [1 for _ in range(self.num_classes)]

        batch_of_images = []

        #iterating the training dataset + loading images
        for i, image in enumerate(os.listdir(os.path.join(path_to_data, self.game,self.size, 'train'))):

            batch_of_images.append([os.path.join(path_to_data, self.game, self.size, 'train', image), class_label])

            if (i+1)%hp.batch_size==0:

                self.train_data.append(batch_of_images)
                batch_of_images = []



        batch_of_images = []

        #iterating the testing dataset + loading images
        for i, image in enumerate(os.listdir(os.path.join(path_to_data, self.game, self.size, 'test'))):



            batch_of_images.append([os.path.join(path_to_data, self.game, self.size, 'test', image), class_label])

            if (i+1)%hp.batch_size==0:
                self.test_data.append(batch_of_images)
                batch_of_images = []


        self.train_data = np.array(self.train_data); self.test_data = np.array(self.test_data)


def parse_arguments():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Enter arguments: --task and --backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--task',
        required = True,
        choices = ['train','test'],
        help = 'specify if running for testing or training'
    )

    parser.add_argument(
        '--backbone',
        required = True,
        choices = ['VGG16','VGG19'],
        help = 'specify if running with backbone of VGG16 or VGG19 for feature extraction'
    )

    parser.add_argument(
        '--atari_game',
        required = True,
        choices = ['MsPacman-v0','BreakoutDeterministic-v4'],
        help = 'specify which game/environment on which WSDDN needs to run'
    )

    parser.add_argument(
        '--load_weights',
        default=None,
        help = 'load h5 file with pretrained weights'
    )


    parser.add_argument(
        '--num_classes',
        required=True,
        help = 'setting up required classes'
    ),

    parser.add_argument(
        '--image_size',
        required=True,
        choices = ['small','large'],
        help = 'select the size of the image to use when training'
    )

    parser.add_argument(
        '--visualize',
        help = 'specify if training losses (validation and train) have to be visualized + saved '
    )

    return parser.parse_args()

def drawBoxes(boxes):
    for (x, y, w, h) in boxes:
        plt.hlines(y, x, x + w)
        plt.hlines(y + h, x, x + w)
        plt.vlines(x, y, y + h)
        plt.vlines(x + w, y, y + h)




def train(model, train_data, val_data, checkpoint_path, logs_path):
    '''
    Runs the main training loop for WSDDN

    train_data contains the image x_batch as well as the image-level labels y_batch

    Note: y_batch is given by a {-1,1}^|C| vector to indicate the presence of classes in image
    '''


    '''Tensorflow function for forward pass'''
    @tf.function
    def train_step(image_batch, label_batch, spatial_reg=False):

        loss_value = 0

        with tf.GradientTape() as tape:

            #iterate all images in each batch
            for image, label in zip(image_batch, label_batch):

                image = tf.expand_dims(image, axis=0)



                output, scores, filtered_origin_rois, spatial_regularizer_output = model.call(image, label, spatial_reg)


                #if no ROIs or regions found, skip to the next image to train on
                if (output, scores, filtered_origin_rois, spatial_regularizer_output)==(None,None,None,None):
                    continue

                loss_value = loss_value + model.crossentropy_loss(tf.expand_dims(output, axis=0), tf.cast(label, tf.float32)) + model.l2_regularizer() + spatial_regularizer_output


            loss_value = loss_value/len(image_batch) #loss averaged over batch

        # pdb.set_trace()
        grads = tape.gradient(loss_value, model.trainable_weights)

        #set the learning rate using the number of iterations for optimizer
        model.optimizer.learning_rate(model.optimizer.iterations)

        #applying gradients with the custom optimizer
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value


    #logging the loss over epochs
    total_loss_train = []
    total_loss_val = []
    min_val_loss = float("inf")

    for epoch in range(hp.num_epochs):

        #record start time per epoch
        start_time = time.time()

        #create progress bar with total number of batches
        progbar = generic_utils.Progbar(len(train_data))

        print('Epoch {}/{}'.format(epoch+1, hp.num_epochs))

        for batch_no, data_batch in enumerate(train_data):


            #shuffle the batch of images first (both images and labels)
            np.random.shuffle(data_batch)

            x_batch = [ np.asarray(Image.open(x[0]).resize(hp.reshaped_size)) for x in data_batch]

            y_batch = list(data_batch[:,1])

            #running training for each image
            train_loss = train_step(x_batch, y_batch, True)
            train_loss = train_loss.numpy()
            pdb.set_trace()

            total_loss_train.append(train_loss)

            #update progress bar on each batch
            progbar.update(batch_no+1, [ ('train_loss', train_loss) ])



            #run model on validation dataset to get validation loss metrics
            if (batch_no+1)%hp.validation_batch_freq==0:

                print('Running validation set...')

                for __, val_data_batch in enumerate(val_data):

                    #shuffle the validation dataset first (both images and labels)
                    np.random.shuffle(val_data_batch)


                    x_val = [ np.asarray(Image.open(x[0]).resize(hp.reshaped_size)) for x in val_data_batch]

                    y_val = list(val_data_batch[:,1])

                    #running training for each image
                    val_loss = train_step(x_val, y_val, True)
                    val_loss = val_loss.numpy()

                    print('val_loss: ', val_loss)

                    #update the minimum loss reference + save the weights files
                    if val_loss < min_val_loss:

                        print('\n Min. validation loss reduced from {} to {}, saving weights'.format(min_val_loss, val_loss))

                        min_val_loss = val_loss
                        model.save_weights(  os.path.join( checkpoint_path, 'epoch_{}_loss{}.hdf5'.format(epoch, val_loss) )  )


                    total_loss_val.append(val_loss)


        end_time = time.time()

        print('Training Loss: ', loss_value)
        print('Validation Loss: ', val_loss)
        print('Elapsed Time: ', end_time-start_time)

        print('Saving logs....')
        #saving log of all losses (training and validation)
        metric_dict = {'train_loss':total_loss_train}
        dataframe = pd.DataFrame(metric_dict)
        dataframe.to_csv(os.path.join(logs_path, 'train_loss_data.csv'))

        metric_dict = {'val_loss':total_loss_val}
        dataframe = pd.DataFrame(metric_dict)
        dataframe.to_csv(os.path.join(logs_path, 'val_loss_data.csv'))

        print('Saved logs!')



    #log and graph the losses using the log_path argument
    plt.plot( range(len(total_loss_train)), total_loss_train )
    plt.title('Training loss by batch')
    plt.xlabel('batch number')
    plt.ylabel('training loss')
    plt.savefig( os.path.join(logs_path, 'train_loss.png') )

    plt.plot( range(len(total_loss_val))  , total_loss_val )
    plt.title('Validation loss by batch')
    plt.xlabel('batch number')
    plt.ylabel('testing loss')
    plt.savefig( os.path.join(logs_path, 'val_loss.png') )

    plot_loss_train = total_loss_train[0:len(total_loss_train):hp.validation_batch_freq]

    plt.plot( range(len(plot_loss_train)), plot_loss_train, label = 'training')
    plt.plot(range(len(plot_loss_val)), plot_loss_val, label = 'validation' )
    plt.title('Training & Validation loss by batch')
    plt.xlabel('batch number')
    plt.ylabel('training/validation loss')
    plt.savefig( os.path.join(logs_path, 'train_val_loss.png') )


    return total_loss_train, total_loss_val

def visualize_losses(train_loss, val_loss, logs_path):

    print('Plotting loss figures....')

    #log and graph the losses using the log_path argument
    plt.plot( len(train_loss), train_loss )
    plt.title('Training loss by batch')
    plt.xlabel('batch number')
    plt.ylabel('training loss')
    plt.savefig( os.path.join(logs_path, 'train_loss.png') )

    plt.plot(  len(val_loss)  , val_loss )
    plt.title('Validation loss by batch')
    plt.xlabel('batch number')
    plt.ylabel('testing loss')
    plt.savefig( os.path.join(logs_path, 'val_loss.png') )

    print('Finished plotting losses!')



def test(model, test_data):
    '''Runs the main testing loop for the WSDDN'''

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)

    pass

def main(ARGS):
    '''Main function parsing input and executing data preprocessing, training + testing'''


    #updating hyperparameters based on number of classes to find
    hp.num_classes = int(ARGS.num_classes)


    if ARGS.load_weights is not None:
        ARGS.load_weights = os.path.abspath(ARGS.load_weights)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_weights).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))


    #extracting data to save the weight checkpoints and logs (for losses or energy function)
    checkpoint_path = os.path.join("weights", str(hp.experiment_number) )
    logs_path = os.path.join("logs" , str(hp.experiment_number) )


    #create WSDDN model instance + pass in backbone architecture to use e.g. VGG16 or VGG19
    model = WSDDN_Model(ARGS.backbone)

    if ARGS.load_weights is not None:
        if ARGS.backbone=='VGG16':
            model.load_weights(ARGS.load_weights, by_name = True)
        elif ARGS.backbone=='VGG19':
            model.load_weights(ARGS.load_weights, by_name = True)


    if ARGS.task=='train' and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if ARGS.task=='train' and not os.path.exists(logs_path):
        os.makedirs(logs_path)


    #add custom metrics and losses to model
    #model.compile(optimizer=model.optimizer, loss=model.loss_fn+model.spatial_regularizer, metrics=[?])


    #collect the dataset into a dictionary list
    data_generator = DatasetCreator(ARGS.atari_game, ARGS.image_size)
    data_generator.create_datasets(PATH_TO_DATA)



    #training or testing the WSDDN model
    if ARGS.task=='test':
        test(model, data_generator.test_data)
    elif ARGS.task=='train':
        total_loss_train, total_loss_val = train(model, data_generator.train_data, data_generator.test_data, checkpoint_path, logs_path)


    if bool(ARGS.visualize):
        visualize_losses(total_loss_train, total_loss_val)



if __name__=='__main__':
    #obtaining args input for training/testing
    ARGS = parse_arguments()

    main(ARGS)
