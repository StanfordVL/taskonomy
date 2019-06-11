INSTANCE_TAG="testing round 2"
COMMAND='echo "Success" > /home/ubuntu/testing.log'

# AMI="ami-660ae31e"
INSTANCE_TYPE="p2.xlarge"
AMI="ami-6a01e812" #extract
# INSTANCE_TYPE="c3.2xlarge"
INSTANCE_COUNT=1
KEY_NAME="taskonomy"
SECURITY_GROUP="launch-wizard-1"
SPOT_PRICE=0.4
ZONE="us-west-2"

START_AT=1
EXIT_AFTER=1

# Now generate transfers
# TASKS="autoencoder colorization denoise edge2d edge3d impainting keypoint2d keypoint3d reshade rgb2depth rgb2mist rgb2sfnorm"
# IGNORED: colorization rgb2mist
TASKS="autoencoder denoise edge2d edge3d impainting keypoint2d keypoint3d reshade rgb2depth rgb2sfnorm"
KEPT_TASKS=$TASKS
# "denoise"

COUNTER=0
for src in $KEPT_TASKS; do
    for dst in $KEPT_TASKS; do
    COUNTER=$[$COUNTER +1]
    if [ "$COUNTER" -lt "$START_AT" ]; then
        echo "Skipping at $COUNTER (starting at $START_AT)"
        continue
    fi
    echo "running $COUNTER"

USER_DATA=$(echo "export INSTANCE_TAG=\"${src}__${dst}__1024 --data-split val\";" | base64)
echo "$USER_DATA" | base64 -d


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

