#!/bin/bash
source /home/ubuntu/.bashrc
dirname="temp"
mkdir -p -- "$dirname"

task_name=""
config_num=""
moment=$(date +"%m%d_%T")
#log_file="train_log_$moment.txt"
log_file="train_log.txt"
resume=""

function usage
{
    echo "-t or --task for task_name; -i or --config for config_num; -l or --log for log_file"
}

while [ "$1" != "" ]
do
   case "$1" in
        -t | --task )          shift
                               task_name=$1
                               ;;
        -i | --config )        shift
                               config_num="config_$1/"
                               ;;
        -l | --log )           shift
                               log_file=$1
                               ;;
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

echo "Running Experiment for Task: $task_name"
echo "Using config: $config_num"
echo "Logging to $log_file"

if [ "$task_name" = "" ]; then
    echo "Task Name is empty..."
    exit 1
fi

config_root="experiments/final"

root_dir="/home/ubuntu/task-taxonomy-331b"
aws_dir="/home/ubuntu/s3/model_log/$task_name"
BUCKET="task-preprocessing-512-oregon/model_log_new"

#if  /usr/bin/aws s3 ls "s3://$BUCKET/" | grep $task_name  ; then
    #if ["$resume" = ""]; then
        #echo "Task Dir, $aws_dir already exists on S3, to resume training, use flag -r, --resume 1"
        #exit  1
    #fi
#fi

#echo $(mkdir -p -- "$aws_dir")
#touch "$aws_dir/CREATE"

task_name=$config_root/$task_name

echo "$(env)" > /home/ubuntu/temp/START_ENV

echo "python -u $root_dir/tools/train.py $root_dir/$task_name/$config_num 2>&1 | tee $root_dir/$task_name/$config_num$log_file" > /home/ubuntu/temp/PY_COMMAND
echo "$root_dir/$task_name/$config_num$log_file" > /home/ubuntu/temp/LOG_FILE
#touch $root_dir/$task_name/$config_num$log_file
/home/ubuntu/anaconda3/bin/python -u $root_dir/tools/train.py $root_dir/$task_name/$config_num 2>&1 | tee $root_dir/$task_name/$config_num$log_file

#python -u train.py $root_dir/$task_name/$config_num 2>&1 | tee $aws_dir/$task_name/$config_num$log_file
