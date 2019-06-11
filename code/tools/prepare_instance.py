import subprocess
import os
def download_task_model(task):
    m_path = os.path.join('/home/ubuntu/s3', "model_log_final", task,
                                     "logs/model.permanent-ckpt") 
    dirs, fname = os.path.split(m_path)
    dst_dir = dirs.replace('/home/ubuntu/s3', "s3://taskonomy-unpacked-oregon")
    tmp_path = "/home/ubuntu/temp/{}".format(task)
    subprocess.call('mkdir -p {}'.format(tmp_path), shell=True)
    tmp_fname = os.path.join(tmp_path, fname)
    aws_cp_command = "aws s3 cp {}.data-00000-of-00001 {}".format(os.path.join(dst_dir, fname), tmp_path)
    subprocess.call(aws_cp_command, shell=True)
    aws_cp_command = "aws s3 cp {}.meta {}".format(os.path.join(dst_dir, fname), tmp_path)
    subprocess.call(aws_cp_command, shell=True)
    aws_cp_command = "aws s3 cp {}.index {}".format(os.path.join(dst_dir, fname), tmp_path)
    subprocess.call(aws_cp_command, shell=True)
    
list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization jigsaw \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_1000 class_places impainting_whole'
list_of_tasks = 'impainting_whole'
list_of_tasks = list_of_tasks.split()
for t in list_of_tasks:
    download_task_model(t)
