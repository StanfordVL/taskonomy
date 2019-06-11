#!/bin/sh
screen -dm bash -c "source /home/ubuntu/.bashrc; sleep 10; /home/ubuntu/task-taxonomy-331b/tools/run_from_task_pool.sh --resume; exec sh"
