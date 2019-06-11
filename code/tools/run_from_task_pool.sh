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
instance_id=$(/usr/bin/ec2metadata --instance-id) 

#ec2addtag $instance_id --region "us-west-2" --aws-access-key $(echo $AWS_ACCESS_KEY_ID) --aws-secret-key $(echo $AWS_SECRET_ACCESS_KEY) --tag Name="$new_tag_name"

cat /home/ubuntu/task-taxonomy-331b/tools/task_list.txt | shuf > order.txt

IFS=$'\r\n' GLOBIGNORE='*' command eval 'TASKS=($(cat order.txt))'
#printf '%s\n' "${TASKS[@]}"
BUCKET="task-preprocessing-512/model_log"

#declare -a TASKS=("curvature_0")

for i in "${TASKS[@]}"
do
    DIR=""
    if ["$resume" = ""]; then
        DIR="s3://$BUCKET/"
        str="$i"
    else
        DIR="s3://$BUCKET/$i/"
        str="RESUME"
    fi
    echo "/usr/bin/aws s3 ls $DIR | grep "$str" "
    
    if /usr/bin/aws s3 ls $DIR | grep "$str" ; then
        echo "DIRECTORY of $i EXISTS, task already been training, run resume.sh if want to resume"
    else
        echo "RUNNING EXPERIMENT: $i"
        /usr/bin/ec2addtag $instance_id --region "us-west-2" --aws-access-key $(echo $AWS_ACCESS_KEY_ID) --aws-secret-key $(echo $AWS_SECRET_ACCESS_KEY) --tag Name="$i"
        
        touch "/home/ubuntu/s3/model_log/$i/RESUME"

        echo "/usr/bin/ec2addtag $instance_id --region us-west-2 --aws-access-key $(echo $AWS_ACCESS_KEY_ID) --aws-secret-key $(echo $AWS_SECRET_ACCESS_KEY) --tag Name=$i" > /home/ubuntu/START_RUN
        if ["$resume" = ""]; then
            /home/ubuntu/task-taxonomy-331b/tools/run.sh -t $i 
        else
            /home/ubuntu/task-taxonomy-331b/tools/run.sh -t $i --resume 
        fi
        /usr/bin/ec2addtag $instance_id --region "us-west-2" --aws-access-key $(echo $AWS_ACCESS_KEY_ID) --aws-secret-key $(echo $AWS_SECRET_ACCESS_KEY) --tag Name="DONE_$i"
        exit 0
        #mkdir /home/ubuntu/s3/model_log/$i
    fi
done

/usr/bin/ec2addtag $instance_id --region "us-west-2" --aws-access-key $(echo $AWS_ACCESS_KEY_ID) --aws-secret-key $(echo $AWS_SECRET_ACCESS_KEY) --tag Name="NOT RUNNING"


