import os
import argparse
import random
import shutil
import pdb



def parse_arguments():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Enter arguments: --task and --backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test_percentage', required=True, help='define percentage (0-1) of data to use in test set')

    parser.add_argument('--atari_game', required=True, choices = ['MsPacman-v0','BreakoutDeterministic-v4'], help = 'specify which game/environment on which WSDDN needs to run')

    return parser.parse_args()

def main(ARGS):
    '''Splits the given dataset (for the ATARI game) between test and training set - as specified by args'''

    test_percentage = float(ARGS.test_percentage)

    game = str(ARGS.atari_game)


    BASE_DATASET_PATH = os.path.abspath('/home/shreyas_sundara_raman/CS2951X_Final_Project/object_detection_networks/data')

    game_path = os.path.join(BASE_DATASET_PATH, game)

    #sample of data indexes to keep in test set
    test_set_sample = len(os.listdir(  os.path.join(game_path, 'small')  ))
    total_num_images = test_set_sample
    test_set_sample = set(random.sample( range(test_set_sample), int(test_set_sample*test_percentage) ))

    print("Splitting dataset ......")

    for image_size in os.listdir(game_path):

        #skip the excel data on each experiment
        if os.path.isfile(os.path.join(game_path,image_size)):
            continue

        # pdb.set_trace()

        game_path_size = os.path.join(game_path, image_size)

        #create the training and testing folder paths if they are not already present
        if not os.path.isdir(os.path.join(game_path_size, 'train')):
            os.makedirs(os.path.join(game_path_size, 'train'))

        if not os.path.isdir(os.path.join(game_path_size, 'test')):
            os.makedirs(os.path.join(game_path_size, 'test'))


        train_path = os.path.join(game_path_size, 'train')
        test_path = os.path.join(game_path_size, 'test')

        # pdb.set_trace()

        for i, image in enumerate( sorted(os.listdir(game_path_size)) ):

            #skip the test images folder from being moved
            if os.path.isdir(image):
                continue

            #move into the test set
            if i in test_set_sample:
                shutil.move( os.path.join(game_path_size, image), test_path )

            #move into the train set
            else:
                shutil.move( os.path.join(game_path_size, image), train_path )



    print("Dataset split successfully!")
    print("Total Images: {} | Train: {} | Test: {}".format(total_num_images, int(total_num_images*(1-test_percentage)), int(total_num_images*(test_percentage)) ))



if __name__=='__main__':

    ARGS = parse_arguments()

    main(ARGS)
