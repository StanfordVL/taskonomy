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
INSTANCE_TYPE="g3.4xlarge"
# INSTANCE_TYPE="g3.4xlarge"
# INSTANCE_TYPE="p3.2xlarge"
#INSTANCE_TYPE="c4.4xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=1.2
ZONE="us-west-2"
SUB_ZONES=( a b c )

# 11 - X
START_AT=2
EXIT_AFTER=500
DATA_USED="16k"


INTERMEDIATE_TASKS="autoencoder__segmentsemantic_rb__denoise__8__unlocked \
impainting_whole__jigsaw__denoise__8__unlocked \
impainting_whole__rgb2depth__denoise__8__unlocked \
autoencoder__rgb2mist__denoise__8__unlocked \
segment2d__rgb2depth__denoise__8__unlocked \
segment2d__room_layout__denoise__8__unlocked \
segment2d__fix_pose__denoise__8__unlocked \
autoencoder__curvature__denoise__8__unlocked \
colorization__edge3d__denoise__8__unlocked \
colorization__segmentsemantic_rb__denoise__8__unlocked \
impainting_whole__segment25d__denoise__8__unlocked \
segment2d__jigsaw__denoise__8__unlocked \
colorization__random__denoise__8__unlocked \
segment2d__rgb2mist__denoise__8__unlocked \
colorization__class_selected__denoise__8__unlocked \
colorization__room_layout__denoise__8__unlocked \
keypoint2d__segmentsemantic_rb__edge2d__8__unlocked \
autoencoder__ego_motion__edge2d__8__unlocked \
autoencoder__random__edge2d__8__unlocked \
keypoint2d__class_1000__edge2d__8__unlocked \
impainting_whole__random__edge2d__8__unlocked \
impainting_whole__vanishing_point_well_defined__edge2d__8__unlocked \
impainting_whole__edge3d__edge2d__8__unlocked \
keypoint2d__class_selected__edge2d__8__unlocked \
segment2d__point_match__edge2d__8__unlocked \
segment2d__segmentsemantic_rb__edge2d__8__unlocked \
autoencoder__jigsaw__edge2d__8__unlocked \
impainting_whole__ego_motion__edge2d__8__unlocked \
segment2d__colorization__edge2d__8__unlocked \
impainting_whole__class_1000__edge2d__8__unlocked \
segment2d__curvature__edge2d__8__unlocked \
segment2d__vanishing_point_well_defined__edge2d__8__unlocked \
denoise__segmentsemantic_rb__autoencoder__8__unlocked \
impainting_whole__jigsaw__autoencoder__8__unlocked \
impainting_whole__rgb2depth__autoencoder__8__unlocked \
denoise__rgb2mist__autoencoder__8__unlocked \
segment2d__rgb2depth__autoencoder__8__unlocked \
segment2d__room_layout__autoencoder__8__unlocked \
segment2d__fix_pose__autoencoder__8__unlocked \
denoise__curvature__autoencoder__8__unlocked \
colorization__reshade__autoencoder__8__unlocked \
colorization__segmentsemantic_rb__autoencoder__8__unlocked \
impainting_whole__segment25d__autoencoder__8__unlocked \
segment2d__jigsaw__autoencoder__8__unlocked \
colorization__random__autoencoder__8__unlocked \
segment2d__rgb2mist__autoencoder__8__unlocked \
colorization__class_selected__autoencoder__8__unlocked \
colorization__room_layout__autoencoder__8__unlocked \
rgb2sfnorm__segmentsemantic_rb__curvature__8__unlocked \
rgb2depth__colorization__curvature__8__unlocked \
rgb2depth__fix_pose__curvature__8__unlocked \
rgb2sfnorm__edge2d__curvature__8__unlocked \
reshade__fix_pose__curvature__8__unlocked \
reshade__point_match__curvature__8__unlocked \
reshade__class_1000__curvature__8__unlocked \
rgb2sfnorm__denoise__curvature__8__unlocked \
rgb2mist__jigsaw__curvature__8__unlocked \
rgb2mist__segmentsemantic_rb__curvature__8__unlocked \
rgb2depth__class_selected__curvature__8__unlocked \
reshade__colorization__curvature__8__unlocked \
rgb2mist__edge3d__curvature__8__unlocked \
reshade__edge2d__curvature__8__unlocked \
rgb2mist__segment25d__curvature__8__unlocked \
rgb2mist__point_match__curvature__8__unlocked \
rgb2sfnorm__segment2d__edge3d__8__unlocked \
keypoint3d__non_fixated_pose__edge3d__8__unlocked \
keypoint3d__class_1000__edge3d__8__unlocked \
rgb2sfnorm__denoise__edge3d__8__unlocked \
curvature__class_1000__edge3d__8__unlocked \
curvature__random__edge3d__8__unlocked \
curvature__fix_pose__edge3d__8__unlocked \
rgb2sfnorm__edge2d__edge3d__8__unlocked \
reshade__point_match__edge3d__8__unlocked \
reshade__segment2d__edge3d__8__unlocked \
keypoint3d__impainting_whole__edge3d__8__unlocked \
curvature__non_fixated_pose__edge3d__8__unlocked \
reshade__rgb2mist__edge3d__8__unlocked \
curvature__denoise__edge3d__8__unlocked \
reshade__segment25d__edge3d__8__unlocked \
reshade__random__edge3d__8__unlocked \
rgb2sfnorm__jigsaw__fix_pose__8__unlocked \
room_layout__edge2d__fix_pose__8__unlocked \
room_layout__colorization__fix_pose__8__unlocked \
rgb2sfnorm__segmentsemantic_rb__fix_pose__8__unlocked \
ego_motion__colorization__fix_pose__8__unlocked \
ego_motion__keypoint2d__fix_pose__8__unlocked \
ego_motion__keypoint3d__fix_pose__8__unlocked \
rgb2sfnorm__curvature__fix_pose__8__unlocked \
rgb2depth__class_1000__fix_pose__8__unlocked \
rgb2depth__jigsaw__fix_pose__8__unlocked \
room_layout__impainting_whole__fix_pose__8__unlocked \
ego_motion__edge2d__fix_pose__8__unlocked \
rgb2depth__reshade__fix_pose__8__unlocked \
ego_motion__segmentsemantic_rb__fix_pose__8__unlocked \
rgb2depth__vanishing_point_well_defined__fix_pose__8__unlocked \
rgb2depth__keypoint2d__fix_pose__8__unlocked \
denoise__point_match__keypoint2d__8__unlocked \
autoencoder__jigsaw__keypoint2d__8__unlocked \
autoencoder__reshade__keypoint2d__8__unlocked \
denoise__rgb2mist__keypoint2d__8__unlocked \
segment2d__reshade__keypoint2d__8__unlocked \
segment2d__room_layout__keypoint2d__8__unlocked \
segment2d__segmentsemantic_rb__keypoint2d__8__unlocked \
denoise__curvature__keypoint2d__8__unlocked \
colorization__non_fixated_pose__keypoint2d__8__unlocked \
colorization__point_match__keypoint2d__8__unlocked \
autoencoder__ego_motion__keypoint2d__8__unlocked \
segment2d__jigsaw__keypoint2d__8__unlocked \
colorization__edge2d__keypoint2d__8__unlocked \
segment2d__rgb2mist__keypoint2d__8__unlocked \
colorization__random__keypoint2d__8__unlocked \
colorization__room_layout__keypoint2d__8__unlocked \
rgb2sfnorm__curvature__ego_motion__8__unlocked \
fix_pose__keypoint2d__ego_motion__8__unlocked \
fix_pose__segment2d__ego_motion__8__unlocked \
rgb2sfnorm__non_fixated_pose__ego_motion__8__unlocked \
room_layout__segment2d__ego_motion__8__unlocked \
room_layout__denoise__ego_motion__8__unlocked \
room_layout__point_match__ego_motion__8__unlocked \
rgb2sfnorm__jigsaw__ego_motion__8__unlocked \
rgb2mist__class_selected__ego_motion__8__unlocked \
rgb2mist__curvature__ego_motion__8__unlocked \
fix_pose__impainting_whole__ego_motion__8__unlocked \
room_layout__keypoint2d__ego_motion__8__unlocked \
rgb2mist__reshade__ego_motion__8__unlocked \
room_layout__non_fixated_pose__ego_motion__8__unlocked \
rgb2mist__vanishing_point_well_defined__ego_motion__8__unlocked \
rgb2mist__denoise__ego_motion__8__unlocked \
curvature__edge2d__keypoint3d__8__unlocked \
rgb2sfnorm__random__keypoint3d__8__unlocked \
rgb2sfnorm__vanishing_point_well_defined__keypoint3d__8__unlocked \
curvature__class_1000__keypoint3d__8__unlocked \
edge3d__vanishing_point_well_defined__keypoint3d__8__unlocked \
edge3d__ego_motion__keypoint3d__8__unlocked \
edge3d__denoise__keypoint3d__8__unlocked \
curvature__autoencoder__keypoint3d__8__unlocked \
reshade__class_selected__keypoint3d__8__unlocked \
reshade__edge2d__keypoint3d__8__unlocked \
rgb2sfnorm__point_match__keypoint3d__8__unlocked \
edge3d__random__keypoint3d__8__unlocked \
reshade__rgb2mist__keypoint3d__8__unlocked \
edge3d__class_1000__keypoint3d__8__unlocked \
reshade__rgb2depth__keypoint3d__8__unlocked \
reshade__ego_motion__keypoint3d__8__unlocked \
edge3d__segment25d__point_match__8__unlocked \
keypoint3d__random__point_match__8__unlocked \
keypoint3d__keypoint2d__point_match__8__unlocked \
edge3d__class_selected__point_match__8__unlocked \
curvature__keypoint2d__point_match__8__unlocked \
curvature__denoise__point_match__8__unlocked \
curvature__colorization__point_match__8__unlocked \
edge3d__class_1000__point_match__8__unlocked \
rgb2sfnorm__autoencoder__point_match__8__unlocked \
rgb2sfnorm__segment25d__point_match__8__unlocked \
keypoint3d__non_fixated_pose__point_match__8__unlocked \
curvature__random__point_match__8__unlocked \
rgb2sfnorm__rgb2depth__point_match__8__unlocked \
curvature__class_selected__point_match__8__unlocked \
rgb2sfnorm__rgb2mist__point_match__8__unlocked \
rgb2sfnorm__denoise__point_match__8__unlocked \
rgb2sfnorm__fix_pose__reshade__8__unlocked \
rgb2mist__non_fixated_pose__reshade__8__unlocked \
rgb2mist__autoencoder__reshade__8__unlocked \
rgb2sfnorm__keypoint2d__reshade__8__unlocked \
edge3d__autoencoder__reshade__8__unlocked \
edge3d__impainting_whole__reshade__8__unlocked \
edge3d__edge2d__reshade__8__unlocked \
rgb2sfnorm__class_1000__reshade__8__unlocked \
keypoint3d__segment2d__reshade__8__unlocked \
keypoint3d__fix_pose__reshade__8__unlocked \
rgb2mist__colorization__reshade__8__unlocked \
edge3d__non_fixated_pose__reshade__8__unlocked \
keypoint3d__curvature__reshade__8__unlocked \
edge3d__keypoint2d__reshade__8__unlocked \
keypoint3d__segment25d__reshade__8__unlocked \
keypoint3d__impainting_whole__reshade__8__unlocked \
rgb2sfnorm__rgb2mist__non_fixated_pose__8__unlocked \
fix_pose__random__non_fixated_pose__8__unlocked \
fix_pose__segment25d__non_fixated_pose__8__unlocked \
rgb2sfnorm__denoise__non_fixated_pose__8__unlocked \
ego_motion__segment25d__non_fixated_pose__8__unlocked \
ego_motion__segment2d__non_fixated_pose__8__unlocked \
ego_motion__point_match__non_fixated_pose__8__unlocked \
rgb2sfnorm__colorization__non_fixated_pose__8__unlocked \
room_layout__jigsaw__non_fixated_pose__8__unlocked \
room_layout__rgb2mist__non_fixated_pose__8__unlocked \
fix_pose__class_selected__non_fixated_pose__8__unlocked \
ego_motion__random__non_fixated_pose__8__unlocked \
room_layout__reshade__non_fixated_pose__8__unlocked \
ego_motion__denoise__non_fixated_pose__8__unlocked \
room_layout__edge3d__non_fixated_pose__8__unlocked \
room_layout__segment2d__non_fixated_pose__8__unlocked \
rgb2mist__fix_pose__rgb2depth__8__unlocked \
reshade__random__rgb2depth__8__unlocked \
reshade__vanishing_point_well_defined__rgb2depth__8__unlocked \
rgb2mist__ego_motion__rgb2depth__8__unlocked \
edge3d__vanishing_point_well_defined__rgb2depth__8__unlocked \
edge3d__impainting_whole__rgb2depth__8__unlocked \
edge3d__class_1000__rgb2depth__8__unlocked \
rgb2mist__class_selected__rgb2depth__8__unlocked \
rgb2sfnorm__edge2d__rgb2depth__8__unlocked \
rgb2sfnorm__fix_pose__rgb2depth__8__unlocked \
reshade__denoise__rgb2depth__8__unlocked \
edge3d__random__rgb2depth__8__unlocked \
rgb2sfnorm__curvature__rgb2depth__8__unlocked \
edge3d__ego_motion__rgb2depth__8__unlocked \
rgb2sfnorm__segment25d__rgb2depth__8__unlocked \
rgb2sfnorm__impainting_whole__rgb2depth__8__unlocked \
rgb2depth__fix_pose__rgb2mist__8__unlocked \
reshade__non_fixated_pose__rgb2mist__8__unlocked \
reshade__vanishing_point_well_defined__rgb2mist__8__unlocked \
rgb2depth__ego_motion__rgb2mist__8__unlocked \
keypoint3d__vanishing_point_well_defined__rgb2mist__8__unlocked \
keypoint3d__impainting_whole__rgb2mist__8__unlocked \
keypoint3d__class_1000__rgb2mist__8__unlocked \
rgb2depth__class_selected__rgb2mist__8__unlocked \
rgb2sfnorm__edge2d__rgb2mist__8__unlocked \
rgb2sfnorm__fix_pose__rgb2mist__8__unlocked \
reshade__autoencoder__rgb2mist__8__unlocked \
keypoint3d__non_fixated_pose__rgb2mist__8__unlocked \
rgb2sfnorm__curvature__rgb2mist__8__unlocked \
keypoint3d__ego_motion__rgb2mist__8__unlocked \
rgb2sfnorm__segment25d__rgb2mist__8__unlocked \
rgb2sfnorm__impainting_whole__rgb2mist__8__unlocked \
reshade__vanishing_point_well_defined__rgb2sfnorm__8__unlocked \
edge3d__non_fixated_pose__rgb2sfnorm__8__unlocked \
edge3d__fix_pose__rgb2sfnorm__8__unlocked \
reshade__autoencoder__rgb2sfnorm__8__unlocked \
curvature__fix_pose__rgb2sfnorm__8__unlocked \
curvature__point_match__rgb2sfnorm__8__unlocked \
curvature__keypoint2d__rgb2sfnorm__8__unlocked \
reshade__segment2d__rgb2sfnorm__8__unlocked \
segment25d__colorization__rgb2sfnorm__8__unlocked \
segment25d__vanishing_point_well_defined__rgb2sfnorm__8__unlocked \
edge3d__class_selected__rgb2sfnorm__8__unlocked \
curvature__non_fixated_pose__rgb2sfnorm__8__unlocked \
segment25d__rgb2mist__rgb2sfnorm__8__unlocked \
curvature__autoencoder__rgb2sfnorm__8__unlocked \
segment25d__rgb2depth__rgb2sfnorm__8__unlocked \
segment25d__point_match__rgb2sfnorm__8__unlocked \
rgb2sfnorm__segment25d__room_layout__8__unlocked \
vanishing_point_well_defined__denoise__room_layout__8__unlocked \
vanishing_point_well_defined__colorization__room_layout__8__unlocked \
rgb2sfnorm__point_match__room_layout__8__unlocked \
edge3d__colorization__room_layout__8__unlocked \
edge3d__autoencoder__room_layout__8__unlocked \
edge3d__fix_pose__room_layout__8__unlocked \
rgb2sfnorm__ego_motion__room_layout__8__unlocked \
rgb2mist__segmentsemantic_rb__room_layout__8__unlocked \
rgb2mist__segment25d__room_layout__8__unlocked \
vanishing_point_well_defined__segment2d__room_layout__8__unlocked \
edge3d__denoise__room_layout__8__unlocked \
rgb2mist__rgb2depth__room_layout__8__unlocked \
edge3d__point_match__room_layout__8__unlocked \
rgb2mist__keypoint3d__room_layout__8__unlocked \
rgb2mist__autoencoder__room_layout__8__unlocked \
keypoint3d__keypoint2d__segment25d__8__unlocked \
rgb2sfnorm__random__segment25d__8__unlocked \
rgb2sfnorm__room_layout__segment25d__8__unlocked \
keypoint3d__segmentsemantic_rb__segment25d__8__unlocked \
edge3d__room_layout__segment25d__8__unlocked \
edge3d__ego_motion__segment25d__8__unlocked \
edge3d__autoencoder__segment25d__8__unlocked \
keypoint3d__denoise__segment25d__8__unlocked \
reshade__class_selected__segment25d__8__unlocked \
reshade__keypoint2d__segment25d__8__unlocked \
rgb2sfnorm__point_match__segment25d__8__unlocked \
edge3d__random__segment25d__8__unlocked \
reshade__rgb2mist__segment25d__8__unlocked \
edge3d__segmentsemantic_rb__segment25d__8__unlocked \
reshade__rgb2depth__segment25d__8__unlocked \
reshade__ego_motion__segment25d__8__unlocked \
curvature__class_1000__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__random__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__point_match__segmentsemantic_rb__8__unlocked \
curvature__edge2d__segmentsemantic_rb__8__unlocked \
reshade__point_match__segmentsemantic_rb__8__unlocked \
reshade__vanishing_point_well_defined__segmentsemantic_rb__8__unlocked \
reshade__colorization__segmentsemantic_rb__8__unlocked \
curvature__segment2d__segmentsemantic_rb__8__unlocked \
edge3d__impainting_whole__segmentsemantic_rb__8__unlocked \
edge3d__class_1000__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__fix_pose__segmentsemantic_rb__8__unlocked \
reshade__random__segmentsemantic_rb__8__unlocked \
edge3d__segment25d__segmentsemantic_rb__8__unlocked \
reshade__edge2d__segmentsemantic_rb__8__unlocked \
edge3d__rgb2mist__segmentsemantic_rb__8__unlocked \
edge3d__vanishing_point_well_defined__segmentsemantic_rb__8__unlocked \
rgb2sfnorm__rgb2mist__vanishing_point_well_defined__8__unlocked \
room_layout__denoise__vanishing_point_well_defined__8__unlocked \
room_layout__impainting_whole__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__ego_motion__vanishing_point_well_defined__8__unlocked \
edge3d__impainting_whole__vanishing_point_well_defined__8__unlocked \
edge3d__non_fixated_pose__vanishing_point_well_defined__8__unlocked \
edge3d__jigsaw__vanishing_point_well_defined__8__unlocked \
rgb2sfnorm__keypoint2d__vanishing_point_well_defined__8__unlocked \
segment25d__class_1000__vanishing_point_well_defined__8__unlocked \
segment25d__rgb2mist__vanishing_point_well_defined__8__unlocked \
room_layout__point_match__vanishing_point_well_defined__8__unlocked \
edge3d__denoise__vanishing_point_well_defined__8__unlocked \
segment25d__keypoint3d__vanishing_point_well_defined__8__unlocked \
edge3d__ego_motion__vanishing_point_well_defined__8__unlocked \
segment25d__curvature__vanishing_point_well_defined__8__unlocked \
segment25d__non_fixated_pose__vanishing_point_well_defined__8__unlocked \
autoencoder__class_selected__segment2d__8__unlocked \
keypoint2d__vanishing_point_well_defined__segment2d__8__unlocked \
keypoint2d__keypoint3d__segment2d__8__unlocked \
autoencoder__rgb2mist__segment2d__8__unlocked \
colorization__keypoint3d__segment2d__8__unlocked \
colorization__room_layout__segment2d__8__unlocked \
colorization__point_match__segment2d__8__unlocked \
autoencoder__class_1000__segment2d__8__unlocked \
edge2d__random__segment2d__8__unlocked \
edge2d__class_selected__segment2d__8__unlocked \
keypoint2d__rgb2depth__segment2d__8__unlocked \
colorization__vanishing_point_well_defined__segment2d__8__unlocked \
edge2d__impainting_whole__segment2d__8__unlocked \
colorization__rgb2mist__segment2d__8__unlocked \
edge2d__segment25d__segment2d__8__unlocked \
edge2d__room_layout__segment2d__8__unlocked \
class_1000__reshade__class_selected__8__unlocked \
segmentsemantic_rb__vanishing_point_well_defined__class_selected__8__unlocked \
segmentsemantic_rb__impainting_whole__class_selected__8__unlocked \
class_1000__segment2d__class_selected__8__unlocked \
edge3d__impainting_whole__class_selected__8__unlocked \
edge3d__rgb2depth__class_selected__8__unlocked \
edge3d__colorization__class_selected__8__unlocked \
class_1000__fix_pose__class_selected__8__unlocked \
keypoint3d__jigsaw__class_selected__8__unlocked \
keypoint3d__reshade__class_selected__8__unlocked \
segmentsemantic_rb__non_fixated_pose__class_selected__8__unlocked \
edge3d__vanishing_point_well_defined__class_selected__8__unlocked \
keypoint3d__point_match__class_selected__8__unlocked \
edge3d__segment2d__class_selected__8__unlocked \
keypoint3d__rgb2mist__class_selected__8__unlocked \
keypoint3d__rgb2depth__class_selected__8__unlocked \
class_selected__reshade__class_1000__8__unlocked \
segmentsemantic_rb__random__class_1000__8__unlocked \
segmentsemantic_rb__autoencoder__class_1000__8__unlocked \
class_selected__colorization__class_1000__8__unlocked \
curvature__autoencoder__class_1000__8__unlocked \
curvature__vanishing_point_well_defined__class_1000__8__unlocked \
curvature__room_layout__class_1000__8__unlocked \
class_selected__fix_pose__class_1000__8__unlocked \
point_match__impainting_whole__class_1000__8__unlocked \
point_match__reshade__class_1000__8__unlocked \
segmentsemantic_rb__ego_motion__class_1000__8__unlocked \
curvature__random__class_1000__8__unlocked \
point_match__keypoint3d__class_1000__8__unlocked \
curvature__colorization__class_1000__8__unlocked \
point_match__rgb2sfnorm__class_1000__8__unlocked \
point_match__vanishing_point_well_defined__class_1000__8__unlocked"



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

export ACTION=TRANSFER;
#export ACTION=EXTRACT_LOSSES;

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




