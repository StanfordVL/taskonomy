#!/bin/bash
resume=""

function usage
{
    echo "--resume, -r for resuming action"
}

while [ "$1" != "" ]
do
   case "$1" in
        -r | --resume)         shift
                               resume=1
                               ;;
        -h | --help )          usage
                               exit
                               ;;
        * )                    usage
                               exit 1
    esac
    shift
done


#/usr/bin/ec2-describe-instance-attribute --user-data --aws-access-key $AWS_ACCESS_KEY
REGION="us-west-2"
INSTANCE_ID=$(/usr/bin/ec2metadata --instance-id)
export INSTANCE_TAG="AUTO-RUNNING"
echo "Credentials:"
echo -e "\tInstance ID: $INSTANCE_ID"
# echo $AWS_SECRET_ACCESS_KEY
# echo $AWS_ACCESS_KEY_ID
# echo $REGION

USER_DATA=$(/usr/bin/ec2-describe-instance-attribute $INSTANCE_ID\
     --aws-access-key $AWS_ACCESS_KEY_ID --aws-secret-key $AWS_SECRET_ACCESS_KEY \
     --user-data\
     --region $REGION | cut -f3 )

# Run all export commands
echo -e "\nRunning export commands first:"
while IFS=";" read -ra ADDR; do
      for cmd in "${ADDR[@]}"; do
          if [[ $cmd == export* ]];
          then
              echo -e "\tCommand: $cmd"
              eval $cmd
          fi
      done
done <<< "$USER_DATA"

echo -e "\nInstance tag: $INSTANCE_TAG"

# Set instance tag
/usr/bin/ec2addtag $INSTANCE_ID --region $REGION --aws-access-key $AWS_ACCESS_KEY_ID \
    --aws-secret-key $AWS_SECRET_ACCESS_KEY --tag Name="$INSTANCE_TAG"

# Run the remaining commands
echo -e "\nRunning user data:"
echo "-----------"
echo "$USER_DATA"
echo "-----------"
echo "$USER_DATA" > /tmp/run_for_startup.sh
source /tmp/run_for_startup.sh

# export INSTANCE_TAG="autoencoder__autoencoder__1024 --action EXTRACT_REPS --data-split train"
# echo "INSTANCE TAG $INSTANCE_TAG"

if [[ ! -z "$INSTANCE_TAG" ]] && [[ ! -z "$ACTION" ]]; then
    bash /home/ubuntu/task-taxonomy-331b/tools/script/generate_all_transfer_configs.sh
    # python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py /home/ubuntu/task-taxonomy-331b/experiments/transfers/${INSTANCE_TAG}
    # python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/transfers --task $INSTANCE_TAG
    if [ "$ACTION" == "EXTRACT_REPS" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments --task $INSTANCE_TAG
    elif [ "$ACTION" == "FINETUNE_IMAGENET_SELECTED" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/train_imagenet.py --transfer --selected --task $INSTANCE_TAG
        python /home/ubuntu/task-taxonomy-331b/tools/extract_imagenet_accuracy.py --transfer --selected --task $INSTANCE_TAG 
    elif [ "$ACTION" == "FINETUNE_PLACES" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/train_places.py --transfer --task $INSTANCE_TAG
        python /home/ubuntu/task-taxonomy-331b/tools/extract_places_accuracy.py --transfer --task $INSTANCE_TAG 
    elif [ "$ACTION" == "FINETUNE_IMAGENET" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/train_imagenet.py --transfer --task $INSTANCE_TAG
        python /home/ubuntu/task-taxonomy-331b/tools/extract_imagenet_accuracy.py --transfer --task $INSTANCE_TAG 
    elif [ "$ACTION" == "EXTRACT_REPS_IMAGENET" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/extract_representations.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments/final --task $INSTANCE_TAG --imagenet
    elif [ "$ACTION" == "SECOND_ORDER_TRANSFER_VIZ" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/run_viz_notebooks_single.py --second-order --t $INSTANCE_TAG   
    elif [ "$ACTION" == "TRANSFER_VIZ" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/run_viz_notebooks_single.py --t $INSTANCE_TAG
    elif [ "$ACTION" == "EXTRACT_LOSSES" ]
    then
        python /home/ubuntu/task-taxonomy-331b/tools/extract_losses.py --cfg_dir /home/ubuntu/task-taxonomy-331b/experiments --task $INSTANCE_TAG
    elif [ "$ACTION" == "TRAIN_TASK_SPEC" ]
    then
        python -u /home/ubuntu/task-taxonomy-331b/tools/train.py /home/ubuntu/task-taxonomy-331b/experiments/final/$INSTANCE_TAG
    elif [ "$ACTION" == "TRAIN_3M" ]
    then
        python -u /home/ubuntu/task-taxonomy-331b/tools/train_3m.py /home/ubuntu/task-taxonomy-331b/experiments/final/$INSTANCE_TAG
    elif [ "$ACTION" == "TRANSFER" ]
    then
        python -u /home/ubuntu/task-taxonomy-331b/tools/transfer.py /home/ubuntu/task-taxonomy-331b/experiments/${INSTANCE_TAG}
    elif [ "$ACTION" == "IMAGENET_KDISTILL" ]
    then
        for split in `seq $((INSTANCE_TAG + 0)) 1 $((INSTANCE_TAG + 1))`; do 
            screen -dm -S instance$split 
            screen -S instance$split -X stuff "bash $(printf \\r)"
            screen -S instance$split -X stuff "source /home/ubuntu/.bashrc; $(printf \\r)"
            screen -S instance$split -X stuff "python /home/ubuntu/task-taxonomy-331b/tools/knowledge_distill_imagenet.py --idx $split && exit"
            screen -S instance$split -X stuff "exit"
        done
        while [ $(screen -ls | wc -l) -ge 4 ]
        do
            echo "Still Running"
            sleep 20s
        done
    elif [ "$ACTION" == "VIDEO" ]
    then
        python -u /home/ubuntu/task-taxonomy-331b/tools/run_viz_video.py --task ${INSTANCE_TAG}
    elif [ "$ACTION" == "SHUTDOWN" ]
    then
        sudo poweroff
    else
        (exit 1)
    fi

    # echo "INSTANCE TAG $INSTANCE_TAG"
    # `python /home/ubuntu/task-taxonomy-331b/tools/script/print_command.py $INSTANCE_TAG`
    # eval(`python /home/ubuntu/task-taxonomy-331b/tools/script/print_command.py $INSTANCE_TAG`)

    OUT=$?
    if [ $OUT -eq 0 ];then
        sudo poweroff
    else
        /usr/bin/ec2addtag $INSTANCE_ID --region $REGION --aws-access-key $AWS_ACCESS_KEY_ID \
        --aws-secret-key $AWS_SECRET_ACCESS_KEY --tag Name="**CRASHED** $INSTANCE_TAG"
        echo "Crashed: leaving on"
    fi
    # python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py --input-task $SRC_TASK --target-task
fi
