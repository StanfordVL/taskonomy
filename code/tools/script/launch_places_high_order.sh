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
AMI="ami-0460b47c" #extract
#INSTANCE_TYPE="p2.xlarge"
INSTANCE_TYPE="g3.4xlarge"
#INSTANCE_TYPE="c3.2xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=2.1
ZONE="us-west-2"
SUB_ZONES=( a b c )

# 11 - X
START_AT=1
EXIT_AFTER=500

# Intermediate tasks left out
# ego_motion
# INTERMEDIATE_TASKS="autoencoder colorization curvature denoise edge2d edge3d \
# fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
# non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
# room_layout segment25d segment2d vanishing_point_well_defined pixels random \
# segmentsemantic_rb class_selected class_1000"

# INTERMEDIATE_TASKS="ego_motion"


IFS=';'
INTERMEDIATE_TASKS="class_1000__class_selected --transfer-type second_order;class_1000__edge3d --transfer-type second_order;class_1000__segmentsemantic_rb --transfer-type second_order;class_1000__curvature --transfer-type second_order;class_selected__edge3d --transfer-type second_order;class_selected__segmentsemantic_rb --transfer-type second_order;class_selected__curvature --transfer-type second_order;edge3d__segmentsemantic_rb --transfer-type second_order;edge3d__curvature --transfer-type second_order;segmentsemantic_rb__curvature --transfer-type second_order;class_1000__class_selected,edge3d --transfer-type third_order;class_1000__class_selected,segmentsemantic_rb --transfer-type third_order;class_1000__class_selected,curvature --transfer-type third_order;class_1000__edge3d,segmentsemantic_rb --transfer-type third_order;class_1000__edge3d,curvature --transfer-type third_order;class_1000__segmentsemantic_rb,curvature --transfer-type third_order;class_selected__edge3d,segmentsemantic_rb --transfer-type third_order;class_selected__edge3d,curvature --transfer-type third_order;class_selected__segmentsemantic_rb,curvature --transfer-type third_order;edge3d__segmentsemantic_rb,curvature --transfer-type third_order;class_1000__class_selected,edge3d,segmentsemantic_rb --transfer-type fourth_order;class_1000__class_selected,edge3d,curvature --transfer-type fourth_order;class_1000__class_selected,edge3d,keypoint3d --transfer-type fourth_order;class_1000__class_selected,segmentsemantic_rb,curvature --transfer-type fourth_order;class_1000__class_selected,segmentsemantic_rb,keypoint3d --transfer-type fourth_order;class_1000__class_selected,curvature,keypoint3d --transfer-type fourth_order;class_1000__edge3d,segmentsemantic_rb,curvature --transfer-type fourth_order;class_1000__edge3d,segmentsemantic_rb,keypoint3d --transfer-type fourth_order;class_1000__edge3d,curvature,keypoint3d --transfer-type fourth_order;class_1000__segmentsemantic_rb,curvature,keypoint3d --transfer-type fourth_order;class_selected__edge3d,segmentsemantic_rb,curvature --transfer-type fourth_order;class_selected__edge3d,segmentsemantic_rb,keypoint3d --transfer-type fourth_order;class_selected__edge3d,curvature,keypoint3d --transfer-type fourth_order;class_selected__segmentsemantic_rb,curvature,keypoint3d --transfer-type fourth_order;edge3d__segmentsemantic_rb,curvature,keypoint3d --transfer-type fourth_order;5 --transfer-type high_order;6 --transfer-type high_order;7 --transfer-type high_order;8 --transfer-type high_order;9 --transfer-type high_order;10 --transfer-type high_order;11 --transfer-type high_order;12 --transfer-type high_order;13 --transfer-type high_order;14 --transfer-type high_order;15 --transfer-type high_order;16 --transfer-type high_order;17 --transfer-type high_order;18 --transfer-type high_order;19 --transfer-type high_order;20 --transfer-type high_order;21 --transfer-type high_order;22 --transfer-type high_order;23 --transfer-type high_order;24 --transfer-type high_order;25 --transfer-type high_order"

# DST tasks left out
# denoise fix_pose point_match class_1000 autoencoder non_fixated_pose rgb2depth segment2d keypoint2d


TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
COUNTER=0
#for src in $SRC_TASKS; do
for intermediate in $INTERMEDIATE_TASKS; do
    sleep 1
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
#export INSTANCE_TAG="final/${intermediate} --data-split val --places";
export INSTANCE_TAG="${intermediate} --metric-only --from-scratch --layer 3 --dropout 0.3 --hidden 2048 --data-use 16000 --alex --num-epochs 120"
#export INSTANCE_TAG="${intermediate} --metric-only --from-scratch --layer 3 --dropout 0.3 --hidden 2048 --data-use 16000 --places-knowledge"
# export INSTANCE_TAG="SASHA P2 WORK INSTANCE";
export HOME="/home/ubuntu"
# export ACTION=SHUTDOWN;
export ACTION=FINETUNE_PLACES;
#export ACTION=EXTRACT_REPS 
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


