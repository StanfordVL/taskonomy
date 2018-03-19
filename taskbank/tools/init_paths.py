import os.path as osp
import sys

cur_dir = osp.dirname( __file__ )
# Search lib first
lib_path = osp.join( cur_dir, '..', 'lib' )
sys.path.insert( 0, lib_path )

# Then elsewhere
root_path = osp.join( cur_dir, '..' )
sys.path.insert( 1, root_path )