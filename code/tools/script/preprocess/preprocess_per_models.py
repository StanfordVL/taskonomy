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
import itertools

# parser = argparse.ArgumentParser(description='Preprocess all tar.gz-ed file into training format')
# parser.add_argument( '--r', dest='root_dir', help='directory containing all tar.gz file' )
# parser.add_argument( '--a', dest='asset_dir', help='directory to store files' )
# parser.add_argument('--combine', dest='combine', action='store_true')
# parser.add_argument('--no-combine', dest='combine', action='store_false')
# parser.set_defaults(combine=False)

def main( _ ):
    print("Starting Preprocessing....")
    args = parser.parse_args()
    root_dir = args.root_dir
    asset_dir = args.asset_dir
    print("Extracting and Processing all tar.gz file located at {root}".format(root=root_dir))
    os.chdir( root_dir )
    print("Changing into directory: {root}".format(root=root_dir))

    all_models = []

    if args.combine:
        for subdir in os.listdir( os.getcwd() ):
            path = os.path.join(os.getcwd(), subdir)
            if os.path.isdir(path):
                all_models.append(subdir)
        print( "found {count} models in root dir {root_dir}".format(
            count=len(all_models), root_dir=root_dir ) )
        single_images_fn = single_images_combine
        nonfixated_view_pairs_fn = nonfixated_view_pairs_combine
        fixated_view_pairs_fn = fixated_view_pairs_combine
        fixated_view_triplets_fn = fixated_view_triplets_combine

    else:
        # Untaring all model files 
        for tar_file in os.listdir( os.getcwd() ):
            result = untar( os.path.join(os.getcwd(), tar_file) )
            if result != "":
                all_models.append(result)
        print( "finished extracting all models, in total {count} models untared".format(count = len(all_models)))
        single_images_fn = single_images
        nonfixated_view_pairs_fn = nonfixated_view_pairs
        fixated_view_pairs_fn = fixated_view_pairs
        fixated_view_triplets_fn = fixated_view_triplets

    # Do the train/val/test split of data
    ratio = {'train': 0.7, 'val': 0.2, 'test': 0.1}
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
    single_images_fn( root_dir, asset_dir, models, unit_size )
    print("Done with Single Images...")

    # Nonfixated Pairs
    nonfixated_view_pairs_fn( root_dir, asset_dir, models, unit_size )
    print("Done with Nonfixated Pairs...")

    # Fixated Pairs
    fixated_view_pairs_fn( root_dir, asset_dir, models, unit_size )
    print("Done with Fixated Pairs...")

    # Fixated Triplets
    fixated_view_triplets_fn( root_dir, asset_dir, models, unit_size )
    print("Done with Fixated Triplets...")

def single_image_per_model( root_dir, modelID, store=False):
    # Process single model for single image filename
    filenames = []
    model_dir = os.path.join(root_dir, modelID, 'rgb')
    for file in os.listdir(model_dir):
        _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
        filenames.append(os.path.join(modelID, point_id, view_id))

    if store:
        fname = '{ID}_image.npy'.format(ID=modelID)
        store_location = os.path.join(root_dir, modelID, fname)
        with open(store_location, 'wb') as store:
            np.save(store, np.asarray(filenames))

    return filenames

def nonfix_pair_per_model( root_dir, modelID, model_thres=0, point_thres=0, store=False):
    # Process single model for nonfix pair
    filenames = []
    model_dir = os.path.join(root_dir, modelID, 'nonfixated')
    for file in os.listdir(model_dir): 
        _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
        full = os.path.join( model_dir, file )
        with open(full) as fp:
            views = json.load(fp)['views']
            length = len(views) 

        comb = [x for x in itertools.combinations(range(length), 2)]
        random.shuffle(comb)
        if point_thres == 0:
            thres = len(comb)
        else:
            thres = point_thres
        for i in range(thres):
            first,second = [views[comb[i][j]] for j in range(2)]
            pair_ij = "{point1},{point2}/{view1},{view2}".format(
                    point1=first['point_uuid'],
                    point2=second['point_uuid'],
                    view1=first['view_id'],
                    view2=second['view_id'])
            filenames.append(os.path.join(modelID,  pair_ij))

    if model_thres > 0:
        random.shuffle(filenames) 
        filenames = filenames[:model_thres]

    if store:
        fname = '{ID}_nonfix_pairs.npy'.format(ID=modelID)
        store_location = os.path.join(root_dir, modelID, fname)
        with open(store_location, 'wb') as store:
            np.save(store, np.asarray(filenames))

    return filenames


def nonfix_trip_per_model( root_dir, modelID, model_thres=0, point_thres=0, store=False):
    # Process single model for nonfix trip
    filenames = []
    model_dir = os.path.join(root_dir, modelID, 'nonfixated')
    for file in os.listdir(model_dir): 
        _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
        full = os.path.join( model_dir, file )
        with open(full) as fp:
            views = json.load(fp)['views']
            length = len(views) 

        comb = [x for x in itertools.combinations(range(length), 3)]
        random.shuffle(comb)
        if point_thres == 0:
            thres = len(comb)
        else:
            thres = point_thres
        for i in range(thres):
            first,second,third = [views[comb[i][j]] for j in range(3)]
            trip = "{p1},{p2},{p3}/{v1},{v2},{v3}".format(
                    p1=first['point_uuid'],
                    p2=second['point_uuid'],
                    p3=third['point_uuid'],
                    v1=first['view_id'],
                    v2=second['view_id'],
                    v3=third['view_id'])
            filenames.append(os.path.join(modelID,  trip))

    if model_thres > 0:
        random.shuffle(filenames) 
        filenames = filenames[:model_thres]

    if store:
        fname = '{ID}_nonfix_trips.npy'.format(ID=modelID)
        store_location = os.path.join(root_dir, modelID, fname)
        with open(store_location, 'wb') as store:
            np.save(store, np.asarray(filenames))

    return filenames

def fix_trip_per_model( root_dir, modelID, point_thres=0, model_thres=0, store=False):
    # Process single model for fix trip
    filenames = []
    model_dir = os.path.join(root_dir, modelID, 'points')
    curr_model_points = {}
    for file in os.listdir(model_dir): 
        _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
        if point_id not in curr_model_points:
            curr_model_points[ point_id ] = 1
        else:
            curr_model_points[ point_id ] = curr_model_points[ point_id ] + 1

    for point_id in curr_model_points.keys():
        length = curr_model_points[point_id]

        comb = [x for x in itertools.combinations(range(length), 3)]
        random.shuffle(comb)
        if point_thres == 0:
            thres = len(comb)
        else:
            thres = point_thres
        for i in range(thres):
            a,b,c = comb[i]
            f = "{point_id}/{i},{j},{k}".format(point_id=point_id, i=a,j=b,k=c)
            filenames.append(os.path.join(modelID,  f))

    if model_thres > 0:
        random.shuffle(filenames) 
        filenames = filenames[:model_thres]

    if store:
        fname = '{ID}_fix_trips.npy'.format(ID=modelID)
        store_location = os.path.join(root_dir, modelID, fname)
        with open(store_location, 'wb') as store:
            np.save(store, np.asarray(filenames))

    return filenames

def fix_pair_per_model( root_dir, modelID, point_thres=0, model_thres=0, store=False):
    # Process single model for fix trip
    filenames = []
    model_dir = os.path.join(root_dir, modelID, 'points')
    curr_model_points = {}
    for file in os.listdir(model_dir): 
        _,point_id,_,view_id,_,_ = file.split('/')[-1].split('_') 
        if point_id not in curr_model_points:
            curr_model_points[ point_id ] = 1
        else:
            curr_model_points[ point_id ] = curr_model_points[ point_id ] + 1
    
    for point_id in curr_model_points.keys():
        length = curr_model_points[point_id]

        comb = [x for x in itertools.combinations(range(length), 2)]
        random.shuffle(comb)
        if point_thres == 0:
            thres = len(comb)
        else:
            thres = point_thres
        for i in range(thres):
            a,b = comb[i]
            f = "{point_id}/{i},{j}".format(point_id=point_id, i=a,j=b)
            filenames.append(os.path.join(modelID,  f))

    if model_thres > 0:
        random.shuffle(filenames) 
        filenames = filenames[:model_thres]

    if store:
        fname = '{ID}_fix_pairs.npy'.format(ID=modelID)
        store_location = os.path.join(root_dir, modelID, fname)
        with open(store_location, 'wb') as store:
            np.save(store, np.asarray(filenames))

    return filenames

def single_images_combine( root_dir, asset_dir, models, unit_size ):
    # Single Images
    for split in ['train', 'val', 'test']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            #model_names = single_image_per_model(root_dir, modelID)
            per_model_file = '{ID}_image.npy'.format(ID=modelID)
            per_model_file = os.path.join(root_dir, modelID, per_model_file)
            print(per_model_file)
            with open(per_model_file, 'r') as fp:
                model_names = np.load(per_model_file)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                random.shuffle(filenames)
                fname = '{split}_image_split_{i}.npy'.format(split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1
        
        random.shuffle(filenames)
        filenames = np.asarray(filenames)
        fname = '{split}_image_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'unit_size' : unit_size, 'filename_list': filename_list}
        
        split_info_location = os.path.join( asset_dir, '{split}_split_image_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def nonfixated_view_pairs_combine( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Non-fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            #model_names = nonfix_pair_per_model( root_dir, modelID )
            per_model_file = '{ID}_nonfix_pairs.npy'.format(ID=modelID)
            per_model_file = os.path.join(root_dir, modelID, per_model_file)
            with open(per_model_file, 'r') as fp:
                model_names = np.load(per_model_file)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                random.shuffle(filenames)
                fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

        filenames = np.asarray(filenames)
        fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_nonfixated_pairs_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )


def nonfixated_view_triplets_combine( root_dir, asset_dir, models, unit_size, per_model_thres=0, threshold=0 ):
    # Non-fixated View Triplet
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            #model_names = nonfix_trip_per_model( root_dir, modelID )
            per_model_file = '{ID}_nonfix_trips.npy'.format(ID=modelID)
            per_model_file = os.path.join(root_dir, modelID, per_model_file)
            with open(per_model_file, 'r') as fp:
                model_names = np.load(per_model_file)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                random.shuffle(filenames)
                fname = '{split}_camera_nonfixated_triplets_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

        filenames = np.asarray(filenames)
        fname = '{split}_camera_nonfixated_triplets_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_nonfixated_triplets_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )  

def fixated_view_pairs_combine( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            #model_names = fix_pair_per_model(root_dir, modelID)
            per_model_file = '{ID}_fix_pairs.npy'.format(ID=modelID)
            per_model_file = os.path.join(root_dir, modelID, per_model_file)
            with open(per_model_file, 'r') as fp:
                model_names = np.load(per_model_file)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                random.shuffle(filenames)
                fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

        random.shuffle(filenames)
        filenames = np.asarray(filenames)
        fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_fixated_pairs_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def nonfixated_view_pairs_split( root_dir, asset_dir, models, unit_size) :

    infos = []
    for split in ['train', 'test', 'val']:
        non_location = os.path.join( root_dir, asset_dir, 'temp', 
                '{split}_camera_nonfixated_pairs_info.pkl'.format(split=split) )
        with open( non_location, 'rb' ) as fp:
            non_info = pickle.load(fp)

        non_f = [ os.path.join(root_dir, asset_dir, 'temp', i) for i in non_info['filename_list']]

        buff = []

        list_len = len(non_f) 
        num_split = 0
        total_size = 0
        filename_list = []
        for i in range(list_len):
            print('Loading {i}'.format(i=non_f[i]))
            non_list = np.load(non_f[i])

            buff.extend(non_list[:])

            random.shuffle(buff)
            while(len(buff) > unit_size):
                fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(
                        split=split, i=num_split)
                print('Saving {i}'.format(i=fname))
                store_location = os.path.join( root_dir, asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(buff[:unit_size]))
                filename_list.append(fname)
                buff = buff[unit_size:]
                num_split = num_split + 1
                total_size += unit_size
                if num_split > 10:
                    break
            if num_split > 10:
                break
           
        if num_split < 10:
            total_size += len(buff)
            random.shuffle(buff)
            buff = np.asarray(buff)
            fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(split=split, i=num_split)
            store_location = os.path.join( root_dir, asset_dir, fname )
            with open(store_location, 'wb') as store:
                np.save(store, buff)
            filename_list.append(fname)

        split_info = {'total_size': total_size, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( root_dir, asset_dir, 
                '{split}_camera_nonfixated_pairs_info.pkl'.format(split=split) )
        with open( split_info_location, "wb" ) as fp: 
            pickle.dump( split_info, fp)

        infos.append(split_info)
    return infos




def matching_pairs_combine( root_dir, asset_dir, unit_size, threshold=0 ):
    # Non-fixated View Pairs
    infos = []
    for split in ['train', 'test', 'val']:
        non_location = os.path.join( root_dir, asset_dir, 
                '{split}_camera_nonfixated_pairs_info.pkl'.format(split=split) )
        fix_location = os.path.join( root_dir, asset_dir, 
                '{split}_camera_fixated_pairs_info.pkl'.format(split=split) )
        with open( non_location, 'rb' ) as fp:
            non_info = pickle.load(fp)

        with open( fix_location, 'rb' ) as fp:
            fix_info = pickle.load(fp)

        infos.append(non_info)
        infos.append(fix_info)
        list_len = min(len(fix_info['filename_list']), len(non_info['filename_list']))

        non_f = [ os.path.join(root_dir, asset_dir, i) for i in non_info['filename_list']]
        fix_f = [ os.path.join(root_dir, asset_dir, i) for i in fix_info['filename_list']]

        filename_list = []
        buff = []
        total_size = 0
        num_split = 0

        for i in range(list_len):
            non_list = np.load(non_f[i])
            fix_list = np.load(fix_f[i])

            length = min(len(non_list), len(fix_list))
            buff.extend(non_list[:length])
            buff.extend(fix_list[:length])

            while(len(buff) > unit_size):
                random.shuffle(buff)
                fname = '{split}_point_match_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( root_dir, asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(buff[:unit_size]))
                filename_list.append(fname)
                buff = buff[unit_size:]
                num_split = num_split + 1
                total_size += unit_size
            
        total_size += len(buff)
        random.shuffle(buff)
        buff = np.asarray(buff)
        fname = '{split}_point_match_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( root_dir, asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, buff)
        filename_list.append(fname)

        split_info = {'total_size': total_size, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( root_dir, asset_dir, 
                '{split}_point_match_info.pkl'.format(split=split) )
        with open( split_info_location, "wb" ) as fp: 
            pickle.dump( split_info, fp)
        infos.append(split_info)
        
    return infos

def fixated_view_triplets_combine( root_dir, asset_dir, models, unit_size, threshold=0 ):
    # Fixated View Pairs
    for split in ['train', 'test', 'val']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            #model_names = fix_trip_per_model(root_dir, modelID)
            per_model_file = '{ID}_fix_trips.npy'.format(ID=modelID)
            per_model_file = os.path.join(root_dir, modelID, per_model_file)
            with open(per_model_file, 'r') as fp:
                model_names = np.load(per_model_file)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                random.shuffle(filenames)
                fname = '{split}_camera_fixated_trips_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

        filenames = np.asarray(filenames)
        fname = '{split}_camera_fixated_trips_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'unit_size' : unit_size, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_fixated_trips_info.pkl'.format(split=split) )
        pickle.dump( split_info, open( split_info_location, "wb" ) )

def single_images( root_dir, asset_dir, models, unit_size ):
    # Single Images
    for split in ['train', 'val', 'test']:
        filename_list = [] 
        filenames = []
        count = 0
        num_split = 0
        for modelID in models[split]:
            model_names = single_image_per_model(root_dir, modelID)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                fname = '{split}_image_split_{i}.npy'.format(split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
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
        for modelID in models[split]:
            model_names = nonfix_pair_per_model( root_dir, modelID )
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                fname = '{split}_camera_nonfixated_pairs_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

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
        for modelID in models[split]:
            model_names = nonfix_trip_per_model( root_dir, modelID )
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                fname = '{split}_camera_nonfixated_triplets_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

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
        for modelID in models[split]:
            model_names = fix_pair_per_model(root_dir, modelID)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                fname = '{split}_camera_fixated_pairs_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

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
        for modelID in models[split]:
            model_names = fix_trip_per_model(root_dir, modelID)
            filenames.extend(model_names)
            count = count + len(model_names)

            while count > unit_size * (num_split + 1):
                #save filenames
                fname = '{split}_camera_fixated_trips_split_{i}.npy'.format(
                        split=split, i=num_split)
                store_location = os.path.join( asset_dir, fname )
                with open(store_location, 'wb') as store:
                    np.save(store, np.asarray(filenames[:unit_size]))
                filename_list.append(fname)
                filenames = filenames[unit_size:]
                num_split = num_split + 1

        filenames = np.asarray(filenames)
        fname = '{split}_camera_fixated_trips_split_{i}.npy'.format(split=split, i=num_split)
        store_location = os.path.join( asset_dir, fname )
        with open(store_location, 'wb') as store:
            np.save(store, filenames)
        filename_list.append(fname)

        split_info = {'total_size': count, 'filename_list': filename_list}
        split_info_location = os.path.join( asset_dir, 
                '{split}_camera_fixated_trips_info.pkl'.format(split=split) )
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
    
# if __name__=='__main__':
    # main( '' )
