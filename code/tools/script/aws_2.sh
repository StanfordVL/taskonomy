#! /bin/bash

for number in {1..20}
do
    echo "Welcome to spot fleet, No.$number... "
    aws ec2 request-spot-instances --spot-price "0.40" --instance-count 1 --launch-specification file://launch_specs_2.json 
    sleep 30
done
