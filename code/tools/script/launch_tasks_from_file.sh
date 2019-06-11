


######################################
# Options to pass to user data
######################################
platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='darwin'
fi

if [[ "$platform" == "linux" ]];
then
    OPTIONS="-w 0"
    ECHO_OPTIONS="-d"
else
    OPTIONS=""
    ECHO_OPTIONS="-D"
fi


# redo 2
START_AT=2
EXIT_AFTER=2
COUNTER=0

while IFS=' ' read -ra ADDR; do
    src="${ADDR[0]}";
    intermediate="${ADDR[1]}";
    dst="${ADDR[2]}";
    echo $src $intermediate $dst;


    # if [ ! -f /home/ubuntu/s3/model_log/representations_transfer_1024/${src}__${dst}__1024_train_representations.pkl ]; then
    #     echo "File not found!";
    # else
    #     continue;
    # fi

    COUNTER=$[$COUNTER +1];
    if [ "$COUNTER" -lt "$START_AT" ]; then
        echo "Skipping at $COUNTER (starting at $START_AT)"
        continue
    fi
    if [ "$COUNTER" -gt "$EXIT_AFTER" ]; then
        echo "EXITING before $COUNTER"
        break
    fi


######################################
# Define user data
######################################
USER_DATA=$(base64 $OPTIONS << END_USER_DATA
export INSTANCE_TAG="${src}__${intermediate}__${dst}__1024";
# export ACTION=TRANSFER;
cd /home/ubuntu/task-taxonomy-331b
git remote add autopull https://alexsax:328d7b8a3e905c1400f293b9c4842fcae3b7dc54@github.com/alexsax/task-taxonomy-331b.git
git pull autopull perceptual-transfer


# export INSTANCE_TAG="${src}__${dst}__1024";
# export INSTANCE_TAG="funnel: ${src}_${intermediate}__${dst}__1024";
export ACTION=SHUTDOWN;
bash /home/ubuntu/task-taxonomy-331b/tools/script/generate_all_transfer_configs.sh

# # Extract direct transfer representations
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${src}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/representations_transfer_1024/ \
#     --data-split train

# # Extract direct funnel representations
# python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py \
#     --task ${src}_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/representations_projection/ \
#     --representation-task /home/ubuntu/s3/model_log/representations/${intermediate}_test_representations.pkl \
#     --data-split train

# # Extract funnel losses
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task ${src}_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/losses_projection_direct/ \
#     --representation-task /home/ubuntu/s3/model_log/representations/${intermediate}_test_representations.pkl \
#     --data-split val

# Extract transitive transfer losses
python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
    --task ${src}__${intermediate}__${dst}__1024 \
    --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
    --out-dir /home/ubuntu/s3/model_log/losses_transfer_transitive/ \
    --data-split val

# Extract transitive funnel losses
# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py \
#     --task all_${intermediate}__${dst}__1024 \
#     --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/ \
#     --out-dir /home/ubuntu/s3/model_log/losses_funnel_transitive/ \
#     --out-name all_${src}__${intermediate}__${dst}__1024_val_representations.pkl \
#     --representation-task /home/ubuntu/s3/model_log/representations_projection/3d_${src}__${intermediate}__1024_val_representations.pkl \
#     --data-split val
  
# export INSTANCE_TAG="${src}__${intermediate}__${dst}__1024";
# export INSTANCE_TAG="${src}";
# export ACTION=SHUTDOWN;
# python /home/ubuntu/task-taxonomy-331b/tools/transfer.py \
#     /home/ubuntu/task-taxonomy-331b/experiments/transfers/${src}__${intermediate}__${dst}__1024/


# python /home/ubuntu/task-taxonomy-331b/tools/get_mean_image.py \
#     /home/ubuntu/task-taxonomy-331b/experiments/aws_second/${src}/ \
#     --stat-type mean \
#     --print-every 100;

# python /home/ubuntu/task-taxonomy-331b/tools/extract_losses_for_avg_image.py \
#     --task ${src} \
#     --avg-type median \
#     --data-split test
END_USER_DATA
);
# echo "$USER_DATA" | base64 $ECHO_OPTIONS;

######################################
# Define AWS instance type parameters
####################################
# AMI="ami-660ae31e"
AMI="ami-26c23e5e" #extract
# INSTANCE_TYPE="p2.xlarge"
# INSTANCE_TYPE="g2.2xlarge"
INSTANCE_TYPE="c3.2xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=0.4001
ZONE="us-west-2"

####################################
# Launch instance
####################################
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
          \"AvailabilityZone\": \"us-west-2a\" \
        } \
    }"
done < /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt
