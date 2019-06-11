#!/bin/bash
declare -a ARCHS=("dilated_l1l2" "dilated_l1l4""dilated_l1l8" "dilated_l3l8" "dilated_l6l2" "dilated_l6l4" "dilated_l6l8" )
for t in "${ARCHS[@]}" 
do
    python /home/ubuntu/task-taxonomy-331b/tools/run_viz_notebooks_single.py --idx -1 --hs 8 --arch $t 2>&1 | tee $t.txt
done
# python /home/ubuntu/task-taxonomy-331b/tools/run_viz_notebooks.py --idx -1 --hs 256
# python /home/ubuntu/task-taxonomy-331b/tools/run_viz_notebooks.py --idx -1 --hs 1024
#git add .
git commit -m 'updated viz notebooks'
#git push autopull perceptual-transfer
