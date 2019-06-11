import sys

SUBSET = sys.argv[1] #'train'
MODEL_SELECTION_FILE = '/cvgl/u/asax/task-taxonomy-331b/assets/data/{0}_models.txt'.format( SUBSET )
IMAGES_LIST_FILE = '/cvgl/u/asax/task-taxonomy-331b/assets/data/all_images.txt'
OUTPUT_IMAGES_LIST_FILE = '/cvgl/u/asax/task-taxonomy-331b/assets/data/{0}_filenames.pkl'.format( SUBSET )

import numpy as np
import pickle as pickle

if __name__ == '__main__':
    valid_models = set()
    with open( MODEL_SELECTION_FILE, 'r' ) as fp:
        for line in fp:
            valid_models.add( line.strip() )

    image_filepaths = []
    with open( IMAGES_LIST_FILE, 'r' ) as fp:
	for line in fp:
	    split_file_path = line.strip().split( "/" )
	    model = split_file_path[0]
	    if model not in valid_models: 
		continue
	    image_filepaths.append( line.strip() )

    with open( OUTPUT_IMAGES_LIST_FILE, 'w' ) as fp:
	pickle.dump( np.array( image_filepaths ), fp )

    print( len( image_filepaths ) )


