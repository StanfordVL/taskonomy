#!/bin/bash
instance_id=$(/usr/bin/ec2metadata --instance-id) 

cat /home/ubuntu/task-taxonomy-331b/tools/task_list.txt | shuf > order.txt

IFS=$'\r\n' GLOBIGNORE='*' command eval 'TASKS=($(cat order.txt))'
#printf '%s\n' "${TASKS[@]}"

for i in "${TASKS[@]}"
do
    IFS='\*' read -ra ADDR <<< "$i"
    new_name="${ADDR[0]}_${ADDR[1]}"
    root="/home/ubuntu/task-taxonomy-331b/experiments/aws_batch"
    mv $root/$i $root/$new_name
done

