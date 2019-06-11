platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='darwin'
fi


# 1-100: g3.4
# 101-200: p2
# AMI="ami-660ae31e"
AMI="ami-d0a16fa8" #extract
AMI="ami-7428ff0c" #extract
#INSTANCE_TYPE="g3.4xlarge"
#INSTANCE_TYPE="p2.xlarge"
# INSTANCE_TYPE="p3.2xlarge"
INSTANCE_TYPE="c4.4xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=1.2
ZONE="us-west-2"
SUB_ZONES=( a b c )

# 11 - X
START_AT=1
EXIT_AFTER=500
DATA_USED="16k"

#echo 'sleeping for 9 hrs...'
#sleep 9h
INTERMEDIATE_TASKS="keypoint2d__autoencoder,denoise,impainting_whole__edge2d__8__unlocked \
keypoint2d__autoencoder,denoise,segment2d__edge2d__8__unlocked \
keypoint2d__autoencoder,denoise,colorization__edge2d__8__unlocked \
keypoint2d__autoencoder,impainting_whole,segment2d__edge2d__8__unlocked \
keypoint2d__autoencoder,impainting_whole,colorization__edge2d__8__unlocked \
keypoint2d__autoencoder,segment2d,colorization__edge2d__8__unlocked \
keypoint2d__denoise,impainting_whole,segment2d__edge2d__8__unlocked \
keypoint2d__denoise,impainting_whole,colorization__edge2d__8__unlocked \
keypoint2d__denoise,segment2d,colorization__edge2d__8__unlocked \
keypoint2d__impainting_whole,segment2d,colorization__edge2d__8__unlocked \
autoencoder__denoise,impainting_whole,segment2d__edge2d__8__unlocked \
autoencoder__denoise,impainting_whole,colorization__edge2d__8__unlocked \
autoencoder__denoise,segment2d,colorization__edge2d__8__unlocked \
autoencoder__impainting_whole,segment2d,colorization__edge2d__8__unlocked \
denoise__impainting_whole,segment2d,colorization__edge2d__8__unlocked \
denoise__impainting_whole,keypoint2d,segment2d__autoencoder__8__unlocked \
denoise__impainting_whole,keypoint2d,colorization__autoencoder__8__unlocked \
denoise__impainting_whole,keypoint2d,random__autoencoder__8__unlocked \
denoise__impainting_whole,segment2d,colorization__autoencoder__8__unlocked \
denoise__impainting_whole,segment2d,random__autoencoder__8__unlocked \
denoise__impainting_whole,colorization,random__autoencoder__8__unlocked \
denoise__keypoint2d,segment2d,colorization__autoencoder__8__unlocked \
denoise__keypoint2d,segment2d,random__autoencoder__8__unlocked \
denoise__keypoint2d,colorization,random__autoencoder__8__unlocked \
denoise__segment2d,colorization,random__autoencoder__8__unlocked \
impainting_whole__keypoint2d,segment2d,colorization__autoencoder__8__unlocked \
impainting_whole__keypoint2d,segment2d,random__autoencoder__8__unlocked \
impainting_whole__keypoint2d,colorization,random__autoencoder__8__unlocked \
impainting_whole__segment2d,colorization,random__autoencoder__8__unlocked \
keypoint2d__segment2d,colorization,random__autoencoder__8__unlocked \
rgb2sfnorm__rgb2depth,keypoint3d,reshade__curvature__8__unlocked \
rgb2sfnorm__rgb2depth,keypoint3d,rgb2mist__curvature__8__unlocked \
rgb2sfnorm__rgb2depth,keypoint3d,edge3d__curvature__8__unlocked \
rgb2sfnorm__rgb2depth,reshade,rgb2mist__curvature__8__unlocked \
rgb2sfnorm__rgb2depth,reshade,edge3d__curvature__8__unlocked \
rgb2sfnorm__rgb2depth,rgb2mist,edge3d__curvature__8__unlocked \
rgb2sfnorm__keypoint3d,reshade,rgb2mist__curvature__8__unlocked \
rgb2sfnorm__keypoint3d,reshade,edge3d__curvature__8__unlocked \
rgb2sfnorm__keypoint3d,rgb2mist,edge3d__curvature__8__unlocked \
rgb2sfnorm__reshade,rgb2mist,edge3d__curvature__8__unlocked \
rgb2depth__keypoint3d,reshade,rgb2mist__curvature__8__unlocked \
rgb2depth__keypoint3d,reshade,edge3d__curvature__8__unlocked \
rgb2depth__keypoint3d,rgb2mist,edge3d__curvature__8__unlocked \
rgb2depth__reshade,rgb2mist,edge3d__curvature__8__unlocked \
keypoint3d__reshade,rgb2mist,edge3d__curvature__8__unlocked \
autoencoder__impainting_whole,keypoint2d,segment2d__denoise__8__unlocked \
autoencoder__impainting_whole,keypoint2d,colorization__denoise__8__unlocked \
autoencoder__impainting_whole,keypoint2d,random__denoise__8__unlocked \
autoencoder__impainting_whole,segment2d,colorization__denoise__8__unlocked \
autoencoder__impainting_whole,segment2d,random__denoise__8__unlocked \
autoencoder__impainting_whole,colorization,random__denoise__8__unlocked \
autoencoder__keypoint2d,segment2d,colorization__denoise__8__unlocked \
autoencoder__keypoint2d,segment2d,random__denoise__8__unlocked \
autoencoder__keypoint2d,colorization,random__denoise__8__unlocked \
autoencoder__segment2d,colorization,random__denoise__8__unlocked \
impainting_whole__keypoint2d,segment2d,colorization__denoise__8__unlocked \
impainting_whole__keypoint2d,segment2d,random__denoise__8__unlocked \
impainting_whole__keypoint2d,colorization,random__denoise__8__unlocked \
impainting_whole__segment2d,colorization,random__denoise__8__unlocked \
keypoint2d__segment2d,colorization,random__denoise__8__unlocked \
rgb2sfnorm__keypoint3d,rgb2depth,curvature__edge3d__8__unlocked \
rgb2sfnorm__keypoint3d,rgb2depth,reshade__edge3d__8__unlocked \
rgb2sfnorm__keypoint3d,rgb2depth,rgb2mist__edge3d__8__unlocked \
rgb2sfnorm__keypoint3d,curvature,reshade__edge3d__8__unlocked \
rgb2sfnorm__keypoint3d,curvature,rgb2mist__edge3d__8__unlocked \
rgb2sfnorm__keypoint3d,reshade,rgb2mist__edge3d__8__unlocked \
rgb2sfnorm__rgb2depth,curvature,reshade__edge3d__8__unlocked \
rgb2sfnorm__rgb2depth,curvature,rgb2mist__edge3d__8__unlocked \
rgb2sfnorm__rgb2depth,reshade,rgb2mist__edge3d__8__unlocked \
rgb2sfnorm__curvature,reshade,rgb2mist__edge3d__8__unlocked \
keypoint3d__rgb2depth,curvature,reshade__edge3d__8__unlocked \
keypoint3d__rgb2depth,curvature,rgb2mist__edge3d__8__unlocked \
keypoint3d__rgb2depth,reshade,rgb2mist__edge3d__8__unlocked \
keypoint3d__curvature,reshade,rgb2mist__edge3d__8__unlocked \
rgb2depth__curvature,reshade,rgb2mist__edge3d__8__unlocked \
denoise__autoencoder,impainting_whole,segment2d__keypoint2d__8__unlocked \
denoise__autoencoder,impainting_whole,colorization__keypoint2d__8__unlocked \
denoise__autoencoder,impainting_whole,edge2d__keypoint2d__8__unlocked \
denoise__autoencoder,segment2d,colorization__keypoint2d__8__unlocked \
denoise__autoencoder,segment2d,edge2d__keypoint2d__8__unlocked \
denoise__autoencoder,colorization,edge2d__keypoint2d__8__unlocked \
denoise__impainting_whole,segment2d,colorization__keypoint2d__8__unlocked \
denoise__impainting_whole,segment2d,edge2d__keypoint2d__8__unlocked \
denoise__impainting_whole,colorization,edge2d__keypoint2d__8__unlocked \
denoise__segment2d,colorization,edge2d__keypoint2d__8__unlocked \
autoencoder__impainting_whole,segment2d,colorization__keypoint2d__8__unlocked \
autoencoder__impainting_whole,segment2d,edge2d__keypoint2d__8__unlocked \
autoencoder__impainting_whole,colorization,edge2d__keypoint2d__8__unlocked \
autoencoder__segment2d,colorization,edge2d__keypoint2d__8__unlocked \
impainting_whole__segment2d,colorization,edge2d__keypoint2d__8__unlocked \
rgb2sfnorm__room_layout,rgb2mist,ego_motion__fix_pose__8__unlocked \
rgb2sfnorm__room_layout,rgb2mist,rgb2depth__fix_pose__8__unlocked \
rgb2sfnorm__room_layout,rgb2mist,reshade__fix_pose__8__unlocked \
rgb2sfnorm__room_layout,ego_motion,rgb2depth__fix_pose__8__unlocked \
rgb2sfnorm__room_layout,ego_motion,reshade__fix_pose__8__unlocked \
rgb2sfnorm__room_layout,rgb2depth,reshade__fix_pose__8__unlocked \
rgb2sfnorm__rgb2mist,ego_motion,rgb2depth__fix_pose__8__unlocked \
rgb2sfnorm__rgb2mist,ego_motion,reshade__fix_pose__8__unlocked \
rgb2sfnorm__rgb2mist,rgb2depth,reshade__fix_pose__8__unlocked \
rgb2sfnorm__ego_motion,rgb2depth,reshade__fix_pose__8__unlocked \
room_layout__rgb2mist,ego_motion,rgb2depth__fix_pose__8__unlocked \
room_layout__rgb2mist,ego_motion,reshade__fix_pose__8__unlocked \
room_layout__rgb2mist,rgb2depth,reshade__fix_pose__8__unlocked \
room_layout__ego_motion,rgb2depth,reshade__fix_pose__8__unlocked \
rgb2mist__ego_motion,rgb2depth,reshade__fix_pose__8__unlocked \
rgb2sfnorm__fix_pose,edge3d,room_layout__ego_motion__8__unlocked \
rgb2sfnorm__fix_pose,edge3d,rgb2mist__ego_motion__8__unlocked \
rgb2sfnorm__fix_pose,edge3d,reshade__ego_motion__8__unlocked \
rgb2sfnorm__fix_pose,room_layout,rgb2mist__ego_motion__8__unlocked \
rgb2sfnorm__fix_pose,room_layout,reshade__ego_motion__8__unlocked \
rgb2sfnorm__fix_pose,rgb2mist,reshade__ego_motion__8__unlocked \
rgb2sfnorm__edge3d,room_layout,rgb2mist__ego_motion__8__unlocked \
rgb2sfnorm__edge3d,room_layout,reshade__ego_motion__8__unlocked \
rgb2sfnorm__edge3d,rgb2mist,reshade__ego_motion__8__unlocked \
rgb2sfnorm__room_layout,rgb2mist,reshade__ego_motion__8__unlocked \
fix_pose__edge3d,room_layout,rgb2mist__ego_motion__8__unlocked \
fix_pose__edge3d,room_layout,reshade__ego_motion__8__unlocked \
fix_pose__edge3d,rgb2mist,reshade__ego_motion__8__unlocked \
fix_pose__room_layout,rgb2mist,reshade__ego_motion__8__unlocked \
edge3d__room_layout,rgb2mist,reshade__ego_motion__8__unlocked \
curvature__rgb2sfnorm,segment25d,edge3d__keypoint3d__8__unlocked \
curvature__rgb2sfnorm,segment25d,reshade__keypoint3d__8__unlocked \
curvature__rgb2sfnorm,segment25d,rgb2mist__keypoint3d__8__unlocked \
curvature__rgb2sfnorm,edge3d,reshade__keypoint3d__8__unlocked \
curvature__rgb2sfnorm,edge3d,rgb2mist__keypoint3d__8__unlocked \
curvature__rgb2sfnorm,reshade,rgb2mist__keypoint3d__8__unlocked \
curvature__segment25d,edge3d,reshade__keypoint3d__8__unlocked \
curvature__segment25d,edge3d,rgb2mist__keypoint3d__8__unlocked \
curvature__segment25d,reshade,rgb2mist__keypoint3d__8__unlocked \
curvature__edge3d,reshade,rgb2mist__keypoint3d__8__unlocked \
rgb2sfnorm__segment25d,edge3d,reshade__keypoint3d__8__unlocked \
rgb2sfnorm__segment25d,edge3d,rgb2mist__keypoint3d__8__unlocked \
rgb2sfnorm__segment25d,reshade,rgb2mist__keypoint3d__8__unlocked \
rgb2sfnorm__edge3d,reshade,rgb2mist__keypoint3d__8__unlocked \
segment25d__edge3d,reshade,rgb2mist__keypoint3d__8__unlocked \
rgb2sfnorm__fix_pose,vanishing_point_well_defined,ego_motion__non_fixated_pose__8__unlocked \
rgb2sfnorm__fix_pose,vanishing_point_well_defined,room_layout__non_fixated_pose__8__unlocked \
rgb2sfnorm__fix_pose,vanishing_point_well_defined,reshade__non_fixated_pose__8__unlocked \
rgb2sfnorm__fix_pose,ego_motion,room_layout__non_fixated_pose__8__unlocked \
rgb2sfnorm__fix_pose,ego_motion,reshade__non_fixated_pose__8__unlocked \
rgb2sfnorm__fix_pose,room_layout,reshade__non_fixated_pose__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,ego_motion,room_layout__non_fixated_pose__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,ego_motion,reshade__non_fixated_pose__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,room_layout,reshade__non_fixated_pose__8__unlocked \
rgb2sfnorm__ego_motion,room_layout,reshade__non_fixated_pose__8__unlocked \
fix_pose__vanishing_point_well_defined,ego_motion,room_layout__non_fixated_pose__8__unlocked \
fix_pose__vanishing_point_well_defined,ego_motion,reshade__non_fixated_pose__8__unlocked \
fix_pose__vanishing_point_well_defined,room_layout,reshade__non_fixated_pose__8__unlocked \
fix_pose__ego_motion,room_layout,reshade__non_fixated_pose__8__unlocked \
vanishing_point_well_defined__ego_motion,room_layout,reshade__non_fixated_pose__8__unlocked \
edge3d__keypoint3d,reshade,curvature__point_match__8__unlocked \
edge3d__keypoint3d,reshade,rgb2sfnorm__point_match__8__unlocked \
edge3d__keypoint3d,reshade,rgb2depth__point_match__8__unlocked \
edge3d__keypoint3d,curvature,rgb2sfnorm__point_match__8__unlocked \
edge3d__keypoint3d,curvature,rgb2depth__point_match__8__unlocked \
edge3d__keypoint3d,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
edge3d__reshade,curvature,rgb2sfnorm__point_match__8__unlocked \
edge3d__reshade,curvature,rgb2depth__point_match__8__unlocked \
edge3d__reshade,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
edge3d__curvature,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
keypoint3d__reshade,curvature,rgb2sfnorm__point_match__8__unlocked \
keypoint3d__reshade,curvature,rgb2depth__point_match__8__unlocked \
keypoint3d__reshade,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
keypoint3d__curvature,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
reshade__curvature,rgb2sfnorm,rgb2depth__point_match__8__unlocked \
rgb2sfnorm__rgb2mist,rgb2depth,edge3d__reshade__8__unlocked \
rgb2sfnorm__rgb2mist,rgb2depth,keypoint3d__reshade__8__unlocked \
rgb2sfnorm__rgb2mist,rgb2depth,curvature__reshade__8__unlocked \
rgb2sfnorm__rgb2mist,edge3d,keypoint3d__reshade__8__unlocked \
rgb2sfnorm__rgb2mist,edge3d,curvature__reshade__8__unlocked \
rgb2sfnorm__rgb2mist,keypoint3d,curvature__reshade__8__unlocked \
rgb2sfnorm__rgb2depth,edge3d,keypoint3d__reshade__8__unlocked \
rgb2sfnorm__rgb2depth,edge3d,curvature__reshade__8__unlocked \
rgb2sfnorm__rgb2depth,keypoint3d,curvature__reshade__8__unlocked \
rgb2sfnorm__edge3d,keypoint3d,curvature__reshade__8__unlocked \
rgb2mist__rgb2depth,edge3d,keypoint3d__reshade__8__unlocked \
rgb2mist__rgb2depth,edge3d,curvature__reshade__8__unlocked \
rgb2mist__rgb2depth,keypoint3d,curvature__reshade__8__unlocked \
rgb2mist__edge3d,keypoint3d,curvature__reshade__8__unlocked \
rgb2depth__edge3d,keypoint3d,curvature__reshade__8__unlocked \
rgb2depth__reshade,edge3d,keypoint3d__rgb2mist__8__unlocked \
rgb2depth__reshade,edge3d,rgb2sfnorm__rgb2mist__8__unlocked \
rgb2depth__reshade,edge3d,curvature__rgb2mist__8__unlocked \
rgb2depth__reshade,keypoint3d,rgb2sfnorm__rgb2mist__8__unlocked \
rgb2depth__reshade,keypoint3d,curvature__rgb2mist__8__unlocked \
rgb2depth__reshade,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
rgb2depth__edge3d,keypoint3d,rgb2sfnorm__rgb2mist__8__unlocked \
rgb2depth__edge3d,keypoint3d,curvature__rgb2mist__8__unlocked \
rgb2depth__edge3d,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
rgb2depth__keypoint3d,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
reshade__edge3d,keypoint3d,rgb2sfnorm__rgb2mist__8__unlocked \
reshade__edge3d,keypoint3d,curvature__rgb2mist__8__unlocked \
reshade__edge3d,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
reshade__keypoint3d,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
edge3d__keypoint3d,rgb2sfnorm,curvature__rgb2mist__8__unlocked \
rgb2mist__reshade,keypoint3d,edge3d__rgb2depth__8__unlocked \
rgb2mist__reshade,keypoint3d,rgb2sfnorm__rgb2depth__8__unlocked \
rgb2mist__reshade,keypoint3d,curvature__rgb2depth__8__unlocked \
rgb2mist__reshade,edge3d,rgb2sfnorm__rgb2depth__8__unlocked \
rgb2mist__reshade,edge3d,curvature__rgb2depth__8__unlocked \
rgb2mist__reshade,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
rgb2mist__keypoint3d,edge3d,rgb2sfnorm__rgb2depth__8__unlocked \
rgb2mist__keypoint3d,edge3d,curvature__rgb2depth__8__unlocked \
rgb2mist__keypoint3d,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
rgb2mist__edge3d,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
reshade__keypoint3d,edge3d,rgb2sfnorm__rgb2depth__8__unlocked \
reshade__keypoint3d,edge3d,curvature__rgb2depth__8__unlocked \
reshade__keypoint3d,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
reshade__edge3d,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
keypoint3d__edge3d,rgb2sfnorm,curvature__rgb2depth__8__unlocked \
reshade__edge3d,keypoint3d,curvature__rgb2sfnorm__8__unlocked \
reshade__edge3d,keypoint3d,segment25d__rgb2sfnorm__8__unlocked \
reshade__edge3d,keypoint3d,rgb2mist__rgb2sfnorm__8__unlocked \
reshade__edge3d,curvature,segment25d__rgb2sfnorm__8__unlocked \
reshade__edge3d,curvature,rgb2mist__rgb2sfnorm__8__unlocked \
reshade__edge3d,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
reshade__keypoint3d,curvature,segment25d__rgb2sfnorm__8__unlocked \
reshade__keypoint3d,curvature,rgb2mist__rgb2sfnorm__8__unlocked \
reshade__keypoint3d,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
reshade__curvature,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
edge3d__keypoint3d,curvature,segment25d__rgb2sfnorm__8__unlocked \
edge3d__keypoint3d,curvature,rgb2mist__rgb2sfnorm__8__unlocked \
edge3d__keypoint3d,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
edge3d__curvature,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
keypoint3d__curvature,segment25d,rgb2mist__rgb2sfnorm__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,reshade,edge3d__room_layout__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,reshade,rgb2mist__room_layout__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,reshade,rgb2depth__room_layout__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,edge3d,rgb2mist__room_layout__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,edge3d,rgb2depth__room_layout__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined,rgb2mist,rgb2depth__room_layout__8__unlocked \
rgb2sfnorm__reshade,edge3d,rgb2mist__room_layout__8__unlocked \
rgb2sfnorm__reshade,edge3d,rgb2depth__room_layout__8__unlocked \
rgb2sfnorm__reshade,rgb2mist,rgb2depth__room_layout__8__unlocked \
rgb2sfnorm__edge3d,rgb2mist,rgb2depth__room_layout__8__unlocked \
vanishing_point_well_defined__reshade,edge3d,rgb2mist__room_layout__8__unlocked \
vanishing_point_well_defined__reshade,edge3d,rgb2depth__room_layout__8__unlocked \
vanishing_point_well_defined__reshade,rgb2mist,rgb2depth__room_layout__8__unlocked \
vanishing_point_well_defined__edge3d,rgb2mist,rgb2depth__room_layout__8__unlocked \
reshade__edge3d,rgb2mist,rgb2depth__room_layout__8__unlocked \
autoencoder__keypoint2d,denoise,colorization__segment2d__8__unlocked \
autoencoder__keypoint2d,denoise,edge2d__segment2d__8__unlocked \
autoencoder__keypoint2d,denoise,impainting_whole__segment2d__8__unlocked \
autoencoder__keypoint2d,colorization,edge2d__segment2d__8__unlocked \
autoencoder__keypoint2d,colorization,impainting_whole__segment2d__8__unlocked \
autoencoder__keypoint2d,edge2d,impainting_whole__segment2d__8__unlocked \
autoencoder__denoise,colorization,edge2d__segment2d__8__unlocked \
autoencoder__denoise,colorization,impainting_whole__segment2d__8__unlocked \
autoencoder__denoise,edge2d,impainting_whole__segment2d__8__unlocked \
autoencoder__colorization,edge2d,impainting_whole__segment2d__8__unlocked \
keypoint2d__denoise,colorization,edge2d__segment2d__8__unlocked \
keypoint2d__denoise,colorization,impainting_whole__segment2d__8__unlocked \
keypoint2d__denoise,edge2d,impainting_whole__segment2d__8__unlocked \
keypoint2d__colorization,edge2d,impainting_whole__segment2d__8__unlocked \
denoise__colorization,edge2d,impainting_whole__segment2d__8__unlocked \
keypoint3d__rgb2sfnorm,curvature,edge3d__segment25d__8__unlocked \
keypoint3d__rgb2sfnorm,curvature,reshade__segment25d__8__unlocked \
keypoint3d__rgb2sfnorm,curvature,rgb2mist__segment25d__8__unlocked \
keypoint3d__rgb2sfnorm,edge3d,reshade__segment25d__8__unlocked \
keypoint3d__rgb2sfnorm,edge3d,rgb2mist__segment25d__8__unlocked \
keypoint3d__rgb2sfnorm,reshade,rgb2mist__segment25d__8__unlocked \
keypoint3d__curvature,edge3d,reshade__segment25d__8__unlocked \
keypoint3d__curvature,edge3d,rgb2mist__segment25d__8__unlocked \
keypoint3d__curvature,reshade,rgb2mist__segment25d__8__unlocked \
keypoint3d__edge3d,reshade,rgb2mist__segment25d__8__unlocked \
rgb2sfnorm__curvature,edge3d,reshade__segment25d__8__unlocked \
rgb2sfnorm__curvature,edge3d,rgb2mist__segment25d__8__unlocked \
rgb2sfnorm__curvature,reshade,rgb2mist__segment25d__8__unlocked \
rgb2sfnorm__edge3d,reshade,rgb2mist__segment25d__8__unlocked \
curvature__edge3d,reshade,rgb2mist__segment25d__8__unlocked \
rgb2sfnorm__room_layout,reshade,edge3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__room_layout,reshade,segment25d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__room_layout,reshade,keypoint3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__room_layout,edge3d,segment25d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__room_layout,edge3d,keypoint3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__room_layout,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__reshade,edge3d,segment25d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__reshade,edge3d,keypoint3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__reshade,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__edge3d,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
room_layout__reshade,edge3d,segment25d__vanishing_point_well_defined__8__unlocked \
room_layout__reshade,edge3d,keypoint3d__vanishing_point_well_defined__8__unlocked \
room_layout__reshade,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
room_layout__edge3d,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
reshade__edge3d,segment25d,keypoint3d__vanishing_point_well_defined__8__unlocked \
curvature__rgb2sfnorm,keypoint3d,reshade__segmentsemantic_rb__8__unlocked \
curvature__rgb2sfnorm,keypoint3d,edge3d__segmentsemantic_rb__8__unlocked \
curvature__rgb2sfnorm,keypoint3d,segment25d__segmentsemantic_rb__8__unlocked \
curvature__rgb2sfnorm,reshade,edge3d__segmentsemantic_rb__8__unlocked \
curvature__rgb2sfnorm,reshade,segment25d__segmentsemantic_rb__8__unlocked \
curvature__rgb2sfnorm,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
curvature__keypoint3d,reshade,edge3d__segmentsemantic_rb__8__unlocked \
curvature__keypoint3d,reshade,segment25d__segmentsemantic_rb__8__unlocked \
curvature__keypoint3d,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
curvature__reshade,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__keypoint3d,reshade,edge3d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__keypoint3d,reshade,segment25d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__keypoint3d,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__reshade,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
keypoint3d__reshade,edge3d,segment25d__segmentsemantic_rb__8__unlocked \
class_selected__segmentsemantic_rb,edge3d,curvature__class_1000__8__unlocked \
class_selected__segmentsemantic_rb,edge3d,point_match__class_1000__8__unlocked \
class_selected__segmentsemantic_rb,edge3d,keypoint3d__class_1000__8__unlocked \
class_selected__segmentsemantic_rb,curvature,point_match__class_1000__8__unlocked \
class_selected__segmentsemantic_rb,curvature,keypoint3d__class_1000__8__unlocked \
class_selected__segmentsemantic_rb,point_match,keypoint3d__class_1000__8__unlocked \
class_selected__edge3d,curvature,point_match__class_1000__8__unlocked \
class_selected__edge3d,curvature,keypoint3d__class_1000__8__unlocked \
class_selected__edge3d,point_match,keypoint3d__class_1000__8__unlocked \
class_selected__curvature,point_match,keypoint3d__class_1000__8__unlocked \
segmentsemantic_rb__edge3d,curvature,point_match__class_1000__8__unlocked \
segmentsemantic_rb__edge3d,curvature,keypoint3d__class_1000__8__unlocked \
segmentsemantic_rb__edge3d,point_match,keypoint3d__class_1000__8__unlocked \
segmentsemantic_rb__curvature,point_match,keypoint3d__class_1000__8__unlocked \
edge3d__curvature,point_match,keypoint3d__class_1000__8__unlocked \
class_1000__segmentsemantic_rb,curvature,edge3d__class_selected__8__unlocked \
class_1000__segmentsemantic_rb,curvature,keypoint3d__class_selected__8__unlocked \
class_1000__segmentsemantic_rb,curvature,point_match__class_selected__8__unlocked \
class_1000__segmentsemantic_rb,edge3d,keypoint3d__class_selected__8__unlocked \
class_1000__segmentsemantic_rb,edge3d,point_match__class_selected__8__unlocked \
class_1000__segmentsemantic_rb,keypoint3d,point_match__class_selected__8__unlocked \
class_1000__curvature,edge3d,keypoint3d__class_selected__8__unlocked \
class_1000__curvature,edge3d,point_match__class_selected__8__unlocked \
class_1000__curvature,keypoint3d,point_match__class_selected__8__unlocked \
class_1000__edge3d,keypoint3d,point_match__class_selected__8__unlocked \
segmentsemantic_rb__curvature,edge3d,keypoint3d__class_selected__8__unlocked \
segmentsemantic_rb__curvature,edge3d,point_match__class_selected__8__unlocked \
segmentsemantic_rb__curvature,keypoint3d,point_match__class_selected__8__unlocked \
segmentsemantic_rb__edge3d,keypoint3d,point_match__class_selected__8__unlocked \
curvature__edge3d,keypoint3d,point_match__class_selected__8__unlocked"

INTERMEDIATE_TASKS="class_1000__class_selected,edge3d,segmentsemantic_rb__class_places__8__unlocked \
class_1000__class_selected,edge3d,curvature__class_places__8__unlocked \
class_1000__class_selected,edge3d,keypoint3d__class_places__8__unlocked \
class_1000__class_selected,segmentsemantic_rb,curvature__class_places__8__unlocked \
class_1000__class_selected,segmentsemantic_rb,keypoint3d__class_places__8__unlocked \
class_1000__class_selected,curvature,keypoint3d__class_places__8__unlocked \
class_1000__edge3d,segmentsemantic_rb,curvature__class_places__8__unlocked \
class_1000__edge3d,segmentsemantic_rb,keypoint3d__class_places__8__unlocked \
class_1000__edge3d,curvature,keypoint3d__class_places__8__unlocked \
class_1000__segmentsemantic_rb,curvature,keypoint3d__class_places__8__unlocked \
class_selected__edge3d,segmentsemantic_rb,curvature__class_places__8__unlocked \
class_selected__edge3d,segmentsemantic_rb,keypoint3d__class_places__8__unlocked \
class_selected__edge3d,curvature,keypoint3d__class_places__8__unlocked \
class_selected__segmentsemantic_rb,curvature,keypoint3d__class_places__8__unlocked \
edge3d__segmentsemantic_rb,curvature,keypoint3d__class_places__8__unlocked"

#INTERMEDIATE_TASKS="rgb2sfnorm__curvature,keypoint3d,class_places__segmentsemantic_rb__8__unlocked \
#rgb2sfnorm__curvature,reshade,class_places__segmentsemantic_rb__8__unlocked \
#rgb2sfnorm__curvature,class_places,edge3d__segmentsemantic_rb__8__unlocked \
#rgb2sfnorm__keypoint3d,reshade,class_places__segmentsemantic_rb__8__unlocked \
#rgb2sfnorm__keypoint3d,class_places,edge3d__segmentsemantic_rb__8__unlocked \
#rgb2sfnorm__reshade,class_places,edge3d__segmentsemantic_rb__8__unlocked \
#curvature__keypoint3d,reshade,class_places__segmentsemantic_rb__8__unlocked \
#curvature__keypoint3d,class_places,edge3d__segmentsemantic_rb__8__unlocked \
#curvature__reshade,class_places,edge3d__segmentsemantic_rb__8__unlocked \
#keypoint3d__reshade,class_places,edge3d__segmentsemantic_rb__8__unlocked"

TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
COUNTER=0
#for src in $SRC_TASKS; do
for intermediate in $INTERMEDIATE_TASKS; do
    for decode in $TARGET_DECODER_FUNCS; do

        COUNTER=$[$COUNTER +1]
        SUB_ZONE=${SUB_ZONES[$((COUNTER%3))]}
        if [ "$COUNTER" -lt "$START_AT" ]; then
            echo "Skipping at $COUNTER (starting at $START_AT)"
            continue
        fi
        echo "running $COUNTER"


        if [[ "$platform" == "linux" ]];
        then
            OPTIONS="-w 0"
            ECHO_OPTIONS="-d"
        else
            OPTIONS=""
            ECHO_OPTIONS="-D"
        fi

                USER_DATA=$(base64 $OPTIONS << END_USER_DATA
export HOME="/home/ubuntu"
export INSTANCE_TAG="fourth_order/${decode}/$DATA_USED/${intermediate}";

#export ACTION=TRANSFER;
export ACTION=EXTRACT_LOSSES;

cd /home/ubuntu/task-taxonomy-331b
git stash
git remote add autopull https://alexsax:328d7b8a3e905c1400f293b9c4842fcae3b7dc54@github.com/alexsax/task-taxonomy-331b.git
git pull autopull perceptual-transfer

watch -n 300 "bash /home/ubuntu/task-taxonomy-331b/tools/script/reboot_if_disconnected.sh" &> /dev/null &

END_USER_DATA
)
                echo "$USER_DATA" | base64 $ECHO_OPTIONS
 
                aws ec2 request-spot-instances \
                --spot-price $SPOT_PRICE \
                --instance-count $INSTANCE_COUNT \
                --region $ZONE \
                --launch-specification \
                    "{ \
                        \"ImageId\":\"$AMI\", \
                        \"InstanceType\":\"$INSTANCE_TYPE\", \
                        \"KeyName\":\"$KEY_NAME\", \
                        \"SecurityGroups\": [\"$SECURITY_GROUP\"], \
                        \"UserData\":\"$USER_DATA\", \
                        \"Placement\": { \
                          \"AvailabilityZone\": \"us-west-2${SUB_ZONE}\" \
                        } \
                    }"
                sleep 1

                        if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
                            echo "EXITING before $COUNTER (after $EXIT_AFTER)"
                            break
                        fi
                        done
                    if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
                        echo "EXITING before $COUNTER (after $EXIT_AFTER)"
                        break
                    fi
                    done 




