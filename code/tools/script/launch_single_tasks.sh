platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='darwin'
fi

# AMI="ami-660ae31e"
INSTANCE_TYPE="p2.xlarge"
AMI="ami-7edd3606" #extract
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=0.4
ZONE="us-west-2"

START_AT=1
EXIT_AFTER=1

# Now generate transfers
TASKS="autoencoder colorization denoise edge2d edge3d impainting keypoint2d keypoint3d reshade rgb2depth rgb2mist rgb2sfnorm colorization vanishing_point"
# IGNORED: colorization rgb2mist
# TASKS="keypoint2d keypoint3d reshade rgb2depth rgb2sfnorm rgb2mist colorization vanishing_point jigsaw"
KEPT_TASKS=$TASKS
# "denoise"

COUNTER=0
for task in $KEPT_TASKS; do
    COUNTER=$[$COUNTER +1]
    if [ "$COUNTER" -lt "$START_AT" ]; then
        echo "Skipping at $COUNTER (starting at $START_AT)"
        continue
    fi

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
export INSTANCE_TAG="${task} --data-split test";
export ACTION=SHUTDOWN;
cd /home/ubuntu/task-taxonomy-331b
git remote add autopull https://alexsax:328d7b8a3e905c1400f293b9c4842fcae3b7dc54@github.com/alexsax/task-taxonomy-331b.git
git pull autopull perceptual-transfer
python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/aws_second/ --task $task --data-split test
END_USER_DATA
)
echo "$USER_DATA" | base64 -D

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
        \"UserData\":\"$USER_DATA\" \
    }"

if [ "$COUNTER" -ge "$EXIT_AFTER" ]; then
    echo "EXITING after $COUNTER"
    break
fi
done

