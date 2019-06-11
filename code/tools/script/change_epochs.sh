#!/bin/bash

cat task_list.txt | shuf > order.txt

IFS=$'\r\n' GLOBIGNORE='*' command eval 'TASKS=($(cat order.txt))'

prefix="cfg.'num.epochs'..=.3"
sub_tring="cfg['num_epochs'] = 12"
echo $prefix

root="/home/ubuntu/task-taxonomy-331b/experiments/aws_batch"
for i in "${TASKS[@]}"
do
    for j in 0 1 2 3  
    do
        dir_name="$root/$i"

        config_file="$dir_name/config.py"
        echo "processing $config_file"
        vim -c ":%s/$prefix/$sub_string/g" -c "wq" $config_file
    done
done


