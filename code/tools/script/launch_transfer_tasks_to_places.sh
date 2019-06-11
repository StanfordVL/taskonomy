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
AMI="ami-7428ff0c" #extract
#INSTANCE_TYPE="p2.xlarge"
INSTANCE_TYPE="g3.4xlarge"
# INSTANCE_TYPE="p3.2xlarge"
# INSTANCE_TYPE="c4.4xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=1.2
ZONE="us-west-2"
SUB_ZONES=( a b c )

# 11 - X
START_AT=1
EXIT_AFTER=500
#DATA_USED="1k 2k 4k 8k 16k 23k"
DATA_USED="1k 16k"

# Intermediate tasks left out
# ego_motion

# SRC_TASKS="autoencoder colorization curvature denoise"

DST_TASKS="class_places"
INTERMEDIATE_TASKS="autoencoder colorization curvature denoise edge2d edge3d \
ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined random \
segmentsemantic_rb class_selected class_1000"

INTERMEDIATE_TASKS="pixels"
# DST tasks left out
# point_match 

# DST_TASKS="autoencoder curvature denoise edge2d edge3d \
# ego_motion fix_pose keypoint2d keypoint3d \
# non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
# room_layout segment25d segment2d vanishing_point_well_defined \
# segmentsemantic_rb class_selected class_1000"



# "segmentsemantic_rb class_selected class_1000"
# "non_fixated_pose"
# "keypoint2d"
# "fix_pose denoise"
# "class_1000 autoencoder rgb2depth segment2d"
# "keypoint3d room_layout"
# "colorization class_selected"

TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
COUNTER=0
#for src in $SRC_TASKS; do
for data in $DATA_USED; do
    for dst in $DST_TASKS; do
        for intermediate in $INTERMEDIATE_TASKS; do
            for decode in $TARGET_DECODER_FUNCS; do
                # if [ "$intermediate" == "$dst" ]; then
                #     echo "Skipping $intermediate==$dst"
                #     continue
                # fi

                # if [ "$src" == "$intermediate" ]; then
                #     echo "Skipping $src==$intermediate"
                #     continue
                # fi
                # if [ "$intermediate" == "$dst" ]; then
                #     echo "Skipping $intermediate==$dst"
                #     continue
                # fi

                COUNTER=$[$COUNTER +1]
                SUB_ZONE=${SUB_ZONES[$((COUNTER%3))]}
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

# export INSTANCE_TAG="full_taskonomy_beta1/${decode}/${data}/${intermediate}__${dst}__8__unlocked";
export INSTANCE_TAG="full_taskonomy_beta1/${decode}/${data}/${intermediate}__${dst}__8__unlocked";

#export INSTANCE_TAG="full_taskonomy_beta1/${decode}/${data}/${intermediate}";
#export INSTANCE_TAG="rep_only_taskonomy/${decode}/${data}/${intermediate}";

#export ACTION=TRANSFER;
export ACTION=EXTRACT_LOSSES;

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
                    if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
                        echo "EXITING before $COUNTER (after $EXIT_AFTER)"
                        break
                    fi
                # echo USER_DATA
                    done 
                if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
                    echo "EXITING after $COUNTER"
                    break
                fi
                done
            if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
                echo "EXITING after $COUNTER"
                break
            fi
            done



# bash /home/ubuntu/task-taxonomy-331b/tools/script/generate_all_transfer_configs.sh
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses_for_avg_image.py \
#     --avg-type mean --data-split test   \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/final/  \
#     --task curvature --print-every 1000
# export INSTANCE_TAG="jigsaw baseline losses";
#  /home/ubuntu/task-taxonomy-331b/tools/extract_downsampled_representations.py --task autoencoder --data-split test
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py --task ${dst} --data-split test
# python /home/ubuntu/task-taxonomy-331b/tools/train.py /home/ubuntu/task-taxonomy-331b/experiments/final/segment2d_estimator_train/

# export INSTANCE_TAG="${src}__${dst}__1024";
# export INSTANCE_TAG="funnel: ${src}_${intermediate}__${dst}__1024";
# export ACTION=SHUTDOWN;
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${dst} \
#     --data-split val
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task ${dst} \
#     --data-split test
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${dst} \
#     --data-split test

# # Extract single-task
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task vanishing_point_well_defined \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/final/ \
#     --data-split val

# # Extract direct transfer representations
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${src}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/representations_transfer_1024/ \
#     --data-split val

# # Extract direct funnel representations
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${src}_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/representations_projection/ \
#     --representation-task /home/ubuntu/s3/model_log/representations/${intermediate}_test_representations.pkl \
#     --data-split val

# # Extract funnel losses
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task ${src}_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/losses_projection_direct/ \
#     --representation-task /home/ubuntu/s3/model_log/representations/${intermediate}_test_representations.pkl \
#     --data-split val

# # Extract transitive transfer losses
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task ${src}__${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/losses_transfer_transitive/ \
#     --data-split val

# Extract transitive funnel losses
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task all_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/losses_funnel_transitive/ \
#     --out-name all_${src}__${intermediate}__${dst}__1024_val_representations.pkl \
#     --representation-task /home/ubuntu/s3/model_log/representations_projection/3d_${src}__${intermediate}__1024_val_representations.pkl \
#     --data-split val
  
# export INSTANCE_TAG="${src}_${intermediate}__${dst}__1024";
# export INSTANCE_TAG="${src}";
# export ACTION=SHUTDOWN;
# bash /home/ubuntu/task-taxonomy-331b/tools/script/generate_all_transfer_configs.sh
# python /home/ubuntu/task-taxonomy-331b/tools/transfer.py /home/ubuntu/task-taxonomy-331b/experiments/transfers/all_to_denoise__1024/
# python /home/ubuntu/task-taxonomy-331b/tools/get_mean_image.py \
#     /home/ubuntu/task-taxonomy-331b/experiments/final/${src}/ \
#     --stat-type mean \
#     --print-every 100;

