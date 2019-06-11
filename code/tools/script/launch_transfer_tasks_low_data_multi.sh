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
AMI="ami-55c2792d" #extract
#INSTANCE_TYPE="p2.xlarge"
INSTANCE_TYPE="g3.4xlarge"
# INSTANCE_TYPE="p3.2xlarge"
# INSTANCE_TYPE="c4.4xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=1.2
ZONE="us-west-2"
SUB_ZONES=( b b b )

# 11 - X
START_AT=1
EXIT_AFTER=500
#DATA_USED="1k 2k 4k 8k 16k 23k"

# Intermediate tasks left out
# ego_motion

INTERMEDIATE_TASKS="rep_only_taskonomy/DO_NOT_REPLACE_TARGET_DECODER/16k/class_selected__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__colorization__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__keypoint2d__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__colorization__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__keypoint2d__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__segment2d__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint2d__colorization__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint2d__segment2d__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/segment2d__colorization__autoencoder__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__reshade__curvature__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__rgb2mist__curvature__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/reshade__rgb2mist__curvature__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2depth__rgb2mist__curvature__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__rgb2mist__curvature__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__colorization__denoise__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__segment2d__denoise__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__colorization__denoise__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__keypoint2d__denoise__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint2d__segment2d__denoise__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__impainting_whole__edge2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__segment2d__edge2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__segment2d__edge2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__segment2d__edge2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__reshade__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__rgb2depth__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2depth__curvature__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__keypoint3d__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__reshade__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__rgb2depth__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__colorization__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__segment2d__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__autoencoder__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__impainting_whole__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__colorization__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/segment2d__colorization__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__reshade__keypoint3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__segment25d__keypoint3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__reshade__keypoint3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/edge3d__autoencoder__reshade__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2depth__keypoint3d__reshade__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2mist__edge3d__reshade__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2mist__rgb2depth__reshade__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__keypoint3d__reshade__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2mist__rgb2sfnorm__rgb2depth__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/reshade__edge3d__rgb2mist__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/reshade__rgb2sfnorm__rgb2mist__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2depth__rgb2sfnorm__rgb2mist__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/edge3d__curvature__rgb2sfnorm__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/edge3d__keypoint3d__rgb2sfnorm__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__curvature__rgb2sfnorm__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/reshade__keypoint3d__rgb2sfnorm__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/reshade__segment25d__rgb2sfnorm__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__reshade__segment25d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint3d__rgb2sfnorm__segment25d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2sfnorm__edge3d__segment25d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__denoise__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__edge2d__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/autoencoder__keypoint2d__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/colorization__edge2d__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__colorization__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/keypoint2d__colorization__segment2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__class_1000__segmentsemantic_rb__8__unlocked"
INTERMEDIATE_TASKS="second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__segment2d__edge2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/rgb2depth__curvature__edge3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/denoise__impainting_whole__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/impainting_whole__colorization__keypoint2d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__segment25d__keypoint3d__8__unlocked \
second_order/DO_NOT_REPLACE_TARGET_DECODER/16k/curvature__class_1000__segmentsemantic_rb__8__unlocked"
COUNTER=0
for intermediate in $INTERMEDIATE_TASKS; do

    COUNTER=$[$COUNTER +1]
    SUB_ZONE=${SUB_ZONES[$((COUNTER%3))]}
#    if [[ "$SUB_ZONE" == "a" ]];
    #then
        #echo "$intermediate"
    #else
        #continue
    #fi
    
    #ONELY=( b c )
    #SUB_ZONE=${ONELY[$((COUNTER%2))]}

    if [ "$COUNTER" -lt "$START_AT" ]; then
        echo "Skipping at $COUNTER (starting at $START_AT)"
        continue
    fi
    echo "running $COUNTER"

# New version take an --action
# USER_DATA=$(echo -n "export INSTANCE_TAG=\"${src}__${dst}__1024 --data-split train --action EXTRACT_LOSSES\";" | base64 -w 0)
# export INSTANCE_TAG="${src}__${dst}__1024 --data-split train";

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

export INSTANCE_TAG="${intermediate}"

export ACTION=TRANSFER;
#export ACTION=EXTRACT_LOSSES;

cd /home/ubuntu/task-taxonomy-331b
git stash
git remote add autopull https://alexsax:328d7b8a3e905c1400f293b9c4842fcae3b7dc54@github.com/alexsax/task-taxonomy-331b.git
git pull autopull perceptual-transfer

watch -n 300 "bash /home/ubuntu/task-taxonomy-331b/tools/script/reboot_if_disconnected.sh" &> /dev/null &
# export INSTANCE_TAG="SASHA G3 WORK INSTANCE 2";
# export INSTANCE_TAG="${SRC_TASKS} -> ${dst}"
# export ACTION=SHUTDOWN;

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


