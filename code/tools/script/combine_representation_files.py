import numpy as np
import pickle as pkl

src = '/home/ubuntu/task-taxonomy-331b/assets/aws_data/train_image_split_0.npy'
dst = '/home/ubuntu/task-taxonomy-331b/assets/aws_data/train_point_match_0.npy'

with open(src, 'rb') as f:
    src_files = np.load(src)

with open(dst, 'rb') as f:
    dst_files = np.load(dst)


def parse_filename( filename ):
    filename = filename.decode("utf-8") 
    model, points, views = filename.split("/") 
    points = points.split(",")
    views = views.split(",")
    return model, points, views

def get_first_file(joined):
    model, points, views = parse_filename(joined)
    return model, points[-1], views[-1]


src_file_to_idx = { get_first_file(joined): i
    for i, joined in enumerate(src_files) }


dst_file_to_src_idx
print(dst_files.shape)


idx = 0
print("File index for {}: {}".format(
    get_first_file(src_files[idx]), 
    idx))

print(get_first_file( src_files[idx] ), 
    src_file_to_idx[ get_first_file( src_files[idx] ) ]
    )