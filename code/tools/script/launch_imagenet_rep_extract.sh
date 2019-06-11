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
AMI="ami-6819d110" #extract
INSTANCE_TYPE="p2.xlarge"
# INSTANCE_TYPE="g3.4xlarge"
#INSTANCE_TYPE="c3.2xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=1.001
ZONE="us-west-2"
SUB_ZONES=( a b c )

# 11 - X
START_AT=1
EXIT_AFTER=50

# Intermediate tasks left out
# ego_motion
# INTERMEDIATE_TASKS="autoencoder colorization curvature denoise edge2d edge3d \
# fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
# non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
# room_layout segment25d segment2d vanishing_point_well_defined pixels random \
# segmentsemantic_rb class_selected class_1000"

# INTERMEDIATE_TASKS="ego_motion"



INTERMEDIATE_TASKS="autoencoder colorization curvature denoise edge2d edge3d \
ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

# DST tasks left out
# denoise fix_pose point_match class_1000 autoencoder non_fixated_pose rgb2depth segment2d keypoint2d


TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
COUNTER=0
#for src in $SRC_TASKS; do
for intermediate in $INTERMEDIATE_TASKS; do
    sleep 2
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
export INSTANCE_TAG="${intermediate}";
# export INSTANCE_TAG="SASHA P2 WORK INSTANCE";
export HOME="/home/ubuntu"
# export ACTION=SHUTDOWN;
export ACTION=EXTRACT_REPS_IMAGENET;
watch -n 300 "bash /home/ubuntu/task-taxonomy-331b/tools/script/reboot_if_disconnected.sh" &> /dev/null &
 
cd /home/ubuntu/task-taxonomy-331b
git stash
git remote add autopull https://alexsax:328d7b8a3e905c1400f293b9c4842fcae3b7dc54@github.com/alexsax/task-taxonomy-331b.git
git pull autopull perceptual-transfer
git pull autopull perceptual-transfer

# bash /home/ubuntu/task-taxonomy-331b/tools/script/generate_all_transfer_configs.sh 
# python /home/ubuntu/task-taxonomy-331b/tools/transfer.py \
#     /home/ubuntu/task-taxonomy-331b/experiments/full_taskonomy_beta1/${decode}/${data}/${intermediate}__${dst}__8__unlocked/

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

    if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
        echo "EXITING before $COUNTER (after $EXIT_AFTER)"
        break
    fi
    done


