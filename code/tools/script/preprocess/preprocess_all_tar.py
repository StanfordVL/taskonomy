from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import tarfile
import math
import random
import numpy as np
import pickle
import json

parser = argparse.ArgumentParser(description='Preprocess all tar.gz-ed file into training format')
parser.add_argument( '--r', dest='root_dir', help='directory containing all tar.gz file' )
parser.add_argument( '--a', dest='asset_dir', help='directory to store files' )


def main( _ ):
    print("Starting Preprocessing....")
    args = parser.parse_args()
    root_dir = args.root_dir
    asset_dir = args.asset_dir
    print("Extracting and Processing all tar.gz file located at {root}".format(root=root_dir))
    os.chdir( root_dir )
    print("Changing into directory: {root}".format(root=root_dir))

    all_models = []
    
    # Untaring all model files 
    # for tar_file in os.listdir( os.getcwd() ):
    #     result = untar( os.path.join(os.getcwd(), tar_file) )
    #     if result != "":
    #         all_models.append(result)
    with open("meta/unprocessed_ids.txt", 'r') as f:
        for l in f.readlines():
            all_models.append(l.rstrip())
    print( "finished tar-ing all models, in total {count} models untared".format(count = len(all_models)))

    # Do the train/val/test split of data
    # train:val:test split has ratio: 0.7,0.2,0.1; with count 87,24,13
    ratio = {'train': 0.805, 'val': 0.128, 'test': 0.00}
    count = {'train': int(math.ceil(ratio['train'] * len(all_models))),
             'val': int(math.floor(ratio['val'] * len(all_models)))}
    count['test'] = len( all_models ) - count['train'] - count['val']

    print( "train:val:test split has ratio: {0},{1},{2}; with count {3},{4},{5}".format(
                                     ratio['train'], ratio['val'], ratio['test'],
                                     count['train'], count['val'], count['test']) )

    random.shuffle(all_models)
    models = {}
    models['train'] = all_models[0:count['train']] 
    models['val'] = all_models[count['train']:][0:count['val']]
    models['test'] = all_models[count['train'] + count['val']:]
    
    model_info_location = os.path.join( asset_dir, 'all_models.pkl' )
    pickle.dump( models, open( model_info_location, "wb" ) )

    unit_size = 20000000
    # Single Images
    single_images( root_dir, asset_dir, models, unit_size )
    print("Done with Single Images...")

    # Nonfixated Pairs
    nonfixated_view_pairs( root_dir, asset_dir, models, unit_size )
    print("Done with Nonfixated Pairs...")

    # Fixated Pairs
    fixated_view_pairs( root_dir, asset_dir, models, unit_size )
    print("Done with Fixated Pairs...")

    # Nonfixated Triplets
    nonfixated_view_triplets( root_dir, asset_dir, models, unit_size )
    print("Done with Nonfixated Triplets...")

def single_images( root_dir, asset_dir, models, unit_size ):
    # Single Images
    for split in ['train', 'val', 'test']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for model_id in models[split]:
            model_dir = os.path.join(root_dir, model_id, 'rgb')
            for file in os.listdir(model_dir):
                _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
                filenames.append(os.path.join(model_id, point_id, view_id))
                count = count + 1
                if count % unit_size == 0:
                    #save filenames
                    filenames = np.asarray(filenames)
                    fname = '{split}_image_split_{i}.npy'.format(split=split, i=num_split)
                    store_location = os.path.join( asset_dir, fname )
                    with open(store_location, 'wb') as store:
                        np.save(store, filenames)
                    filename_list.append(fname)
                    filenames = []
                    num_split = num_split + 1

        
        filenames = np.asarray(filenames)
        fname = '{split}_image_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        
        split_info_location = os.path.join( asset_dir, '{split}_split_image_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def nonfixated_view_pairs( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Non-fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for model_id in models[split]:
            model_dir = os.path.join(root_dir, model_id, 'nonfixated')
            for file in os.listdir(model_dir): 
                _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
                full = os.path.join( model_dir, file )
                with open(full) as fp:
                    views = json.load(fp)['views']
                    length = len(views) 
                for i in range(length):
                    for j in range(i+1,length):
                        first = views[i]
                        second = views[j]
                        pair_ij = "{point1},{point2}/{view1},{view2}".format(
                                point1=first['point_uuid'],
                                point2=second['point_uuid'],
                                view1=first['view_id'],
                                view2=second['view_id'])
                        filenames.append(os.path.join(model_id,  pair_ij))
                        count = count + 1
                        if count % unit_size == 0:
                            #save filenames
                            filenames = np.asarray(filenames)
                            fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(
                                    split=split, i=num_split)
                            store_location = os.path.join( asset_dir, fname )
                            with open(store_location, 'wb') as store:
                                np.save(store, filenames)
                            filename_list.append(fname)
                            filenames = []
                            num_split = num_split + 1
                            if count > threshold and threshold > 0:
                                return

        filenames = np.asarray(filenames)
        fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_nonfixated_pairs_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )
 

def nonfixated_view_triplets( root_dir, asset_dir, models, unit_size, per_model_thres=0, threshold=0 ):
    # Non-fixated View Triplet
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for model_id in models[split]:
            model_count = 0
            model_dir = os.path.join(root_dir, model_id, 'nonfixated')
            for file in os.listdir(model_dir): 
                print(count)
                _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
                full = os.path.join( model_dir, file )
                with open(full) as fp:
                    views = json.load(fp)['views']
                    length = len(views) 
                for i in range(length):
                    for j in range(i+1,length):
                        for k in range(j+1, length):
                            first = views[i]
                            second = views[j]
                            third = views[k]
                            trip = "{p1},{p2},{p3}/{v1},{v2},{v3}".format(
                                    p1=first['point_uuid'],
                                    p2=second['point_uuid'],
                                    p3=third['point_uuid'],
                                    v1=first['view_id'],
                                    v2=second['view_id'],
                                    v3=third['view_id'])
                            filenames.append(os.path.join(model_id,  trip))
                            count = count + 1
                            model_count = model_count + 1

                            if count % unit_size == 0:
                                #save filenames
                                filenames = np.asarray(filenames)
                                fname = '{split}_camera_nonfixated_triplets_split_{i}.npy'.format(
                                        split=split, i=num_split)
                                store_location = os.path.join( asset_dir, fname )
                                with open(store_location, 'wb') as store:
                                    np.save(store, filenames)
                                filename_list.append(fname)
                                filenames = []
                                num_split = num_split + 1
                                if count > threshold and threshold > 0:
                                    return

                            if model_count > per_model_thres and per_model_thres > 0:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break


        filenames = np.asarray(filenames)
        fname = '{split}_camera_nonfixated_triplets_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_nonfixated_triplets_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )  

def fixated_view_pairs( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for model_id in models[split]:
            model_dir = os.path.join(root_dir, model_id, 'points')
            curr_model_points = {}
            for file in os.listdir(model_dir): 
                _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
                if point_id not in curr_model_points:
                    curr_model_points[ point_id ] = 0
                else:
                    curr_model_points[ point_id ] = curr_model_points[ point_id ] + 1
            
            for point_id in curr_model_points.keys():
                length = curr_model_points[point_id] + 1
                for i in range(length):
                    for j in range(i+1,length):
                        f = "{point_id}/{i},{j}".format(point_id=point_id, i=i,j=j)
                        filenames.append(os.path.join(model_id,  f))
                        count = count + 1
                        if count % unit_size == 0:
                            #save filenames
                            filenames = np.asarray(filenames)
                            fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(
                                    split=split, i=num_split)
                            store_location = os.path.join( asset_dir, fname )
                            with open(store_location, 'wb') as store:
                                np.save(store, filenames)
                            filename_list.append(fname)
                            filenames = []
                            num_split = num_split + 1
                            if count > threshold and threshold > 0:
                                return

        filenames = np.asarray(filenames)
        fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_fixated_pairs_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def fixated_view_triplets( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for model_id in models[split]:
            model_dir = os.path.join(root_dir, model_id, 'points')
            curr_model_points = {}
            for file in os.listdir(model_dir): 
                _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
                if point_id not in curr_model_points:
                    curr_model_points[ point_id ] = 0
                else:
                    curr_model_points[ point_id ] = curr_model_points[ point_id ] + 1
            
            for point_id in curr_model_points.keys():
                length = curr_model_points[point_id] + 1
                for i in range(length):
                    for j in range(i+1,length):
                        for k in range(j+1, length):
                            f = "{point_id}/{i},{j},{k}".format(point_id=point_id, i=i,j=j,k=k)
                            filenames.append(os.path.join(model_id,  f))
                            count = count + 1
                            if count % unit_size == 0:
                                #save filenames
                                filenames = np.asarray(filenames)
                                fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(
                                        split=split, i=num_split)
                                store_location = os.path.join( asset_dir, fname )
                                with open(store_location, 'wb') as store:
                                    np.save(store, filenames)
                                filename_list.append(fname)
                                filenames = []
                                num_split = num_split + 1
                                if count > threshold and threshold > 0:
                                    return

        filenames = np.asarray(filenames)
        fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_fixated_pairs_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def untar( fname ):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        modelID = strip_modelID( fname )
        print( "Extracted model tar ball {model_name}".format(model_name=modelID) )
        return modelID
    print( "Not extracting file: {f}".format(f=fname) )
    return ""

def strip_modelID( fname ):
    components = fname.split('.')
    modelID = components[-3].split('/')[-1]
    return modelID  
    
if __name__=='__main__':
    main( '' )
