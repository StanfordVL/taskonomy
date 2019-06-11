#!/bin/bash

cat task_list.txt | shuf > order.txt

IFS=$'\r\n' GLOBIGNORE='*' command eval 'TASKS=($(cat order.txt))'

declare -a lr=( "1e-5" "1e-6" "1e-4" "1e-3" )

prefix="cfg['initial_learning_rate'] = "
echo $prefix

root="/home/ubuntu/task-taxonomy-331b/experiments/aws_batch"
for i in "${TASKS[@]}"
do
    for j in 0 1 2 3  
    do
        src_dir="$root/$i"
        dir_name="$root/$i*$j"
        echo "cp -r $src_dir $dir_name"
        #cp -r $src_dir $dir_name

        lr_insert="$prefix${lr[j]}"

        config_file="$dir_name/config.py"
        #vim -c ":%s/1e-6, 1e-1/&\r    $lr_insert/g" -c "wq" $config_file
    done
done

for i in "${TASKS[@]}"
do
    dir="$root/$i"
    rm -r $dir
done

