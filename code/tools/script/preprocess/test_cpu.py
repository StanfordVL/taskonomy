import sys
sys.path.append('/home/ubuntu/task-taxonomy-331b/lib')
import os
import data.load_ops as lp
import data.task_data_loading as dp
import numpy as np
import threading
import tensorflow as tf
import os, sys, time
import concurrent.futures

def main( _ ):
    root_dir = '/home/ubuntu/task-taxonomy-331b'
    meta_file = 'train_image_split_0.npy'
    images = np.load(os.path.join(root_dir, 'assets/aws_data', meta_file))
    
    bucket_dir = '/home/ubuntu/s3'
    test_image = os.path.join(bucket_dir, images[3].decode('UTF-8'))
    print(test_image)
    
    # Test image name extension
    full_image_path = dp.make_filename_for_domain( test_image, 'rgb')
    print(full_image_path)

    # Test Image loading
    import time

    start = time.time()
    image_data = lp.load_raw_image( full_image_path )

    end = time.time()
    print("Loading a image from S3 takes:")
    print(end - start)

    repeat = 10
    list_names = [full_image_path] * repeat 
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=repeat) as executor:
        executor.map(lp.load_raw_image, list_names )
    end = time.time()
    print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    print(end - start)

    repeat = 50
    list_names = [full_image_path] * repeat 
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=repeat) as executor:
        executor.map(lp.load_raw_image, list_names )
    end = time.time()
    print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    print(end - start)
   #  repeat = 1
    # start = time.time()
    # pool = GreenPool(size=repeat)
    # for i in range(repeat):
        # pool.spawn_n(lp.load_from_aws, full_image_path)
    # pool.waitall()
    # end = time.time()
    # print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    # print(end - start)


    # repeat = 10
    # start = time.time()
    # pool = GreenPool(size=repeat)
    # for i in range(repeat):
        # pool.spawn_n(lp.load_from_aws, full_image_path)
    # pool.waitall()
    # end = time.time()
    # print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    # print(end - start)


    # repeat = 20
    # start = time.time()
    # pool = GreenPool(size=repeat)
    # for i in range(repeat):
        # pool.spawn_n(lp.load_from_aws, full_image_path)
    # pool.waitall()
    # end = time.time()
    # print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    # print(end - start)

    # repeat = 30
    # start = time.time()
    # pool = GreenPool(size=repeat)
    # for i in range(repeat):
        # pool.spawn_n(lp.load_from_aws, full_image_path)
    # pool.waitall()
    # end = time.time()
    # print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    # print(end - start)

    # repeat = 100
    # start = time.time()
    # pool = GreenPool(size=repeat)
    # for i in range(repeat):
        # pool.spawn_n(lp.load_from_aws, full_image_path)
    # pool.waitall()
    # end = time.time()
    # print("Downloading {repeat} times using {repeat} threads..".format(repeat=repeat))
    # print(end - start)


if __name__=='__main__':
    main( '' )
