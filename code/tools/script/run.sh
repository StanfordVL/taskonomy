#! /bin/bash
dirname="temp"
mkdir -p -- "$dirname"

task_name=""
config_num=""
moment=$(date +"%m%d_%T")
#log_file="train_log_$moment.txt"
log_file="train_log.txt"

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

root_dir="../experiments"
aws_dir="/home/ubuntu/s3/model_log"

#touch $root_dir/$task_name/$config_num$log_file
python -u train.py $root_dir/$task_name/$config_num 2>&1 | tee $root_dir/$task_name/$config_num$log_file

#python -u train.py $root_dir/$task_name/$config_num 2>&1 | tee $aws_dir/$task_name/$config_num$log_file
