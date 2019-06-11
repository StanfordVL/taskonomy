import subprocess 
import os
import pickle
import concurrent.futures

def change_name( model_id ):
    s3_model_dir = 's3://task-preprocessing-512/{ID}'.format(ID=model_id)
    shell_command = 'aws s3 --recursive mv {dir}/keypoint_2d {dir}/keypoint2d'.format(dir=s3_model_dir)
    result = subprocess.check_output(shell_command, shell=True)

def main( _ ):
    with open('/home/ubuntu/task-taxonomy-331b/assets/aws_data/all_models.pkl', 'rb') as fp:
        data = pickle.load(fp)

    full_list = data['train'] + data['val'] + data['test']
    
    #change_name(full_list[0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        result = executor.map(change_name, full_list[1:])



if __name__=='__main__':
    main( '' )
