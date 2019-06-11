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
#AMI="ami-7428ff0c" #extract
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

#echo 'sleeping for 4 hrs...'
#sleep 4h
#INTERMEDIATE_TASKS="denoise__impainting_whole__autoencoder__8__unlocked"
INTERMEDIATE_TASKS="keypoint2d__autoencoder__edge2d__8__unlocked \
keypoint2d__denoise__edge2d__8__unlocked \
keypoint2d__impainting_whole__edge2d__8__unlocked \
keypoint2d__segment2d__edge2d__8__unlocked \
autoencoder__denoise__edge2d__8__unlocked \
autoencoder__impainting_whole__edge2d__8__unlocked \
autoencoder__segment2d__edge2d__8__unlocked \
denoise__impainting_whole__edge2d__8__unlocked \
denoise__segment2d__edge2d__8__unlocked \
impainting_whole__segment2d__edge2d__8__unlocked \
denoise__impainting_whole__autoencoder__8__unlocked \
denoise__keypoint2d__autoencoder__8__unlocked \
denoise__segment2d__autoencoder__8__unlocked \
denoise__colorization__autoencoder__8__unlocked \
impainting_whole__keypoint2d__autoencoder__8__unlocked \
impainting_whole__segment2d__autoencoder__8__unlocked \
impainting_whole__colorization__autoencoder__8__unlocked \
keypoint2d__segment2d__autoencoder__8__unlocked \
keypoint2d__colorization__autoencoder__8__unlocked \
segment2d__colorization__autoencoder__8__unlocked \
rgb2sfnorm__rgb2depth__curvature__8__unlocked \
rgb2sfnorm__keypoint3d__curvature__8__unlocked \
rgb2sfnorm__reshade__curvature__8__unlocked \
rgb2sfnorm__rgb2mist__curvature__8__unlocked \
rgb2depth__keypoint3d__curvature__8__unlocked \
rgb2depth__reshade__curvature__8__unlocked \
rgb2depth__rgb2mist__curvature__8__unlocked \
keypoint3d__reshade__curvature__8__unlocked \
keypoint3d__rgb2mist__curvature__8__unlocked \
reshade__rgb2mist__curvature__8__unlocked \
autoencoder__impainting_whole__denoise__8__unlocked \
autoencoder__keypoint2d__denoise__8__unlocked \
autoencoder__segment2d__denoise__8__unlocked \
autoencoder__colorization__denoise__8__unlocked \
impainting_whole__keypoint2d__denoise__8__unlocked \
impainting_whole__segment2d__denoise__8__unlocked \
impainting_whole__colorization__denoise__8__unlocked \
keypoint2d__segment2d__denoise__8__unlocked \
keypoint2d__colorization__denoise__8__unlocked \
segment2d__colorization__denoise__8__unlocked \
denoise__autoencoder__keypoint2d__8__unlocked \
denoise__impainting_whole__keypoint2d__8__unlocked \
denoise__segment2d__keypoint2d__8__unlocked \
denoise__colorization__keypoint2d__8__unlocked \
autoencoder__impainting_whole__keypoint2d__8__unlocked \
autoencoder__segment2d__keypoint2d__8__unlocked \
autoencoder__colorization__keypoint2d__8__unlocked \
impainting_whole__segment2d__keypoint2d__8__unlocked \
impainting_whole__colorization__keypoint2d__8__unlocked \
segment2d__colorization__keypoint2d__8__unlocked \
rgb2sfnorm__fix_pose__ego_motion__8__unlocked \
rgb2sfnorm__edge3d__ego_motion__8__unlocked \
rgb2sfnorm__room_layout__ego_motion__8__unlocked \
rgb2sfnorm__rgb2mist__ego_motion__8__unlocked \
fix_pose__edge3d__ego_motion__8__unlocked \
fix_pose__room_layout__ego_motion__8__unlocked \
fix_pose__rgb2mist__ego_motion__8__unlocked \
edge3d__room_layout__ego_motion__8__unlocked \
edge3d__rgb2mist__ego_motion__8__unlocked \
room_layout__rgb2mist__ego_motion__8__unlocked \
rgb2sfnorm__keypoint3d__edge3d__8__unlocked \
rgb2sfnorm__rgb2depth__edge3d__8__unlocked \
rgb2sfnorm__curvature__edge3d__8__unlocked \
rgb2sfnorm__reshade__edge3d__8__unlocked \
keypoint3d__rgb2depth__edge3d__8__unlocked \
keypoint3d__curvature__edge3d__8__unlocked \
keypoint3d__reshade__edge3d__8__unlocked \
rgb2depth__curvature__edge3d__8__unlocked \
rgb2depth__reshade__edge3d__8__unlocked \
curvature__reshade__edge3d__8__unlocked \
rgb2sfnorm__room_layout__fix_pose__8__unlocked \
rgb2sfnorm__rgb2mist__fix_pose__8__unlocked \
rgb2sfnorm__ego_motion__fix_pose__8__unlocked \
rgb2sfnorm__rgb2depth__fix_pose__8__unlocked \
room_layout__rgb2mist__fix_pose__8__unlocked \
room_layout__ego_motion__fix_pose__8__unlocked \
room_layout__rgb2depth__fix_pose__8__unlocked \
rgb2mist__ego_motion__fix_pose__8__unlocked \
rgb2mist__rgb2depth__fix_pose__8__unlocked \
ego_motion__rgb2depth__fix_pose__8__unlocked \
rgb2sfnorm__fix_pose__non_fixated_pose__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined__non_fixated_pose__8__unlocked \
rgb2sfnorm__ego_motion__non_fixated_pose__8__unlocked \
rgb2sfnorm__room_layout__non_fixated_pose__8__unlocked \
fix_pose__vanishing_point_well_defined__non_fixated_pose__8__unlocked \
fix_pose__ego_motion__non_fixated_pose__8__unlocked \
fix_pose__room_layout__non_fixated_pose__8__unlocked \
vanishing_point_well_defined__ego_motion__non_fixated_pose__8__unlocked \
vanishing_point_well_defined__room_layout__non_fixated_pose__8__unlocked \
ego_motion__room_layout__non_fixated_pose__8__unlocked \
edge3d__keypoint3d__point_match__8__unlocked \
edge3d__reshade__point_match__8__unlocked \
edge3d__curvature__point_match__8__unlocked \
edge3d__rgb2sfnorm__point_match__8__unlocked \
keypoint3d__reshade__point_match__8__unlocked \
keypoint3d__curvature__point_match__8__unlocked \
keypoint3d__rgb2sfnorm__point_match__8__unlocked \
reshade__curvature__point_match__8__unlocked \
reshade__rgb2sfnorm__point_match__8__unlocked \
curvature__rgb2sfnorm__point_match__8__unlocked \
curvature__rgb2sfnorm__keypoint3d__8__unlocked \
curvature__segment25d__keypoint3d__8__unlocked \
curvature__edge3d__keypoint3d__8__unlocked \
curvature__reshade__keypoint3d__8__unlocked \
rgb2sfnorm__segment25d__keypoint3d__8__unlocked \
rgb2sfnorm__edge3d__keypoint3d__8__unlocked \
rgb2sfnorm__reshade__keypoint3d__8__unlocked \
segment25d__edge3d__keypoint3d__8__unlocked \
segment25d__reshade__keypoint3d__8__unlocked \
edge3d__reshade__keypoint3d__8__unlocked \
rgb2sfnorm__rgb2mist__reshade__8__unlocked \
rgb2sfnorm__rgb2depth__reshade__8__unlocked \
rgb2sfnorm__edge3d__reshade__8__unlocked \
rgb2sfnorm__keypoint3d__reshade__8__unlocked \
rgb2mist__rgb2depth__reshade__8__unlocked \
rgb2mist__edge3d__reshade__8__unlocked \
rgb2mist__keypoint3d__reshade__8__unlocked \
rgb2depth__edge3d__reshade__8__unlocked \
rgb2depth__keypoint3d__reshade__8__unlocked \
edge3d__keypoint3d__reshade__8__unlocked \
rgb2mist__reshade__rgb2depth__8__unlocked \
rgb2mist__keypoint3d__rgb2depth__8__unlocked \
rgb2mist__edge3d__rgb2depth__8__unlocked \
rgb2mist__rgb2sfnorm__rgb2depth__8__unlocked \
reshade__keypoint3d__rgb2depth__8__unlocked \
reshade__edge3d__rgb2depth__8__unlocked \
reshade__rgb2sfnorm__rgb2depth__8__unlocked \
keypoint3d__edge3d__rgb2depth__8__unlocked \
keypoint3d__rgb2sfnorm__rgb2depth__8__unlocked \
edge3d__rgb2sfnorm__rgb2depth__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined__room_layout__8__unlocked \
rgb2sfnorm__reshade__room_layout__8__unlocked \
rgb2sfnorm__edge3d__room_layout__8__unlocked \
rgb2sfnorm__rgb2mist__room_layout__8__unlocked \
vanishing_point_well_defined__reshade__room_layout__8__unlocked \
vanishing_point_well_defined__edge3d__room_layout__8__unlocked \
vanishing_point_well_defined__rgb2mist__room_layout__8__unlocked \
reshade__edge3d__room_layout__8__unlocked \
reshade__rgb2mist__room_layout__8__unlocked \
edge3d__rgb2mist__room_layout__8__unlocked \
rgb2depth__reshade__rgb2mist__8__unlocked \
rgb2depth__edge3d__rgb2mist__8__unlocked \
rgb2depth__keypoint3d__rgb2mist__8__unlocked \
rgb2depth__rgb2sfnorm__rgb2mist__8__unlocked \
reshade__edge3d__rgb2mist__8__unlocked \
reshade__keypoint3d__rgb2mist__8__unlocked \
reshade__rgb2sfnorm__rgb2mist__8__unlocked \
edge3d__keypoint3d__rgb2mist__8__unlocked \
edge3d__rgb2sfnorm__rgb2mist__8__unlocked \
keypoint3d__rgb2sfnorm__rgb2mist__8__unlocked \
reshade__edge3d__rgb2sfnorm__8__unlocked \
reshade__keypoint3d__rgb2sfnorm__8__unlocked \
reshade__curvature__rgb2sfnorm__8__unlocked \
reshade__segment25d__rgb2sfnorm__8__unlocked \
edge3d__keypoint3d__rgb2sfnorm__8__unlocked \
edge3d__curvature__rgb2sfnorm__8__unlocked \
edge3d__segment25d__rgb2sfnorm__8__unlocked \
keypoint3d__curvature__rgb2sfnorm__8__unlocked \
keypoint3d__segment25d__rgb2sfnorm__8__unlocked \
curvature__segment25d__rgb2sfnorm__8__unlocked \
keypoint3d__rgb2sfnorm__segment25d__8__unlocked \
keypoint3d__curvature__segment25d__8__unlocked \
keypoint3d__edge3d__segment25d__8__unlocked \
keypoint3d__reshade__segment25d__8__unlocked \
rgb2sfnorm__curvature__segment25d__8__unlocked \
rgb2sfnorm__edge3d__segment25d__8__unlocked \
rgb2sfnorm__reshade__segment25d__8__unlocked \
curvature__edge3d__segment25d__8__unlocked \
curvature__reshade__segment25d__8__unlocked \
edge3d__reshade__segment25d__8__unlocked \
autoencoder__keypoint2d__segment2d__8__unlocked \
autoencoder__denoise__segment2d__8__unlocked \
autoencoder__colorization__segment2d__8__unlocked \
autoencoder__edge2d__segment2d__8__unlocked \
keypoint2d__denoise__segment2d__8__unlocked \
keypoint2d__colorization__segment2d__8__unlocked \
keypoint2d__edge2d__segment2d__8__unlocked \
denoise__colorization__segment2d__8__unlocked \
denoise__edge2d__segment2d__8__unlocked \
colorization__edge2d__segment2d__8__unlocked \
curvature__rgb2sfnorm__segmentsemantic_rb__8__unlocked \
curvature__keypoint3d__segmentsemantic_rb__8__unlocked \
curvature__reshade__segmentsemantic_rb__8__unlocked \
curvature__edge3d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__keypoint3d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__reshade__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__edge3d__segmentsemantic_rb__8__unlocked \
keypoint3d__reshade__segmentsemantic_rb__8__unlocked \
keypoint3d__edge3d__segmentsemantic_rb__8__unlocked \
reshade__edge3d__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__room_layout__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__reshade__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__edge3d__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__segment25d__vanishing_point_well_defined__8__unlocked \
room_layout__reshade__vanishing_point_well_defined__8__unlocked \
room_layout__edge3d__vanishing_point_well_defined__8__unlocked \
room_layout__segment25d__vanishing_point_well_defined__8__unlocked \
reshade__edge3d__vanishing_point_well_defined__8__unlocked \
reshade__segment25d__vanishing_point_well_defined__8__unlocked \
edge3d__segment25d__vanishing_point_well_defined__8__unlocked \
class_1000__segmentsemantic_rb__class_selected__8__unlocked \
class_1000__curvature__class_selected__8__unlocked \
class_1000__edge3d__class_selected__8__unlocked \
class_1000__keypoint3d__class_selected__8__unlocked \
segmentsemantic_rb__curvature__class_selected__8__unlocked \
segmentsemantic_rb__edge3d__class_selected__8__unlocked \
segmentsemantic_rb__keypoint3d__class_selected__8__unlocked \
curvature__edge3d__class_selected__8__unlocked \
curvature__keypoint3d__class_selected__8__unlocked \
edge3d__keypoint3d__class_selected__8__unlocked \
class_selected__segmentsemantic_rb__class_1000__8__unlocked \
class_selected__edge3d__class_1000__8__unlocked \
class_selected__curvature__class_1000__8__unlocked \
class_selected__point_match__class_1000__8__unlocked \
segmentsemantic_rb__edge3d__class_1000__8__unlocked \
segmentsemantic_rb__curvature__class_1000__8__unlocked \
segmentsemantic_rb__point_match__class_1000__8__unlocked \
edge3d__curvature__class_1000__8__unlocked \
edge3d__point_match__class_1000__8__unlocked \
curvature__point_match__class_1000__8__unlocked"

INTERMEDIATE_TASKS="class_1000__class_selected__class_places__8__unlocked \
class_1000__edge3d__class_places__8__unlocked \
class_1000__segmentsemantic_rb__class_places__8__unlocked \
class_1000__curvature__class_places__8__unlocked \
class_selected__edge3d__class_places__8__unlocked \
class_selected__segmentsemantic_rb__class_places__8__unlocked \
class_selected__curvature__class_places__8__unlocked \
edge3d__segmentsemantic_rb__class_places__8__unlocked \
edge3d__curvature__class_places__8__unlocked \
segmentsemantic_rb__curvature__class_places__8__unlocked"
INTERMEDIATE_TASKS="rgb2sfnorm__class_places__segmentsemantic_rb__8__unlocked \
curvature__class_places__segmentsemantic_rb__8__unlocked \
keypoint3d__class_places__segmentsemantic_rb__8__unlocked \
reshade__class_places__segmentsemantic_rb__8__unlocked"
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
export INSTANCE_TAG="second_order/${decode}/$DATA_USED/${intermediate}";

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




