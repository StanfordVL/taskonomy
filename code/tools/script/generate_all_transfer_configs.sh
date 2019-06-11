#!/bin/bash
TRANSFER_COMMAND_FILE=/tmp/run_all_transfers.txt
rm $TRANSFER_COMMAND_FILE
touch $TRANSFER_COMMAND_FILE
echo "Generating config files..."

###################
# RESNET26  Task  #
###################
SRC_TASKS="denoise_25 rgb2sfnorm_25 random \
class_places_25 room_layout_25"

TASKS="denoise_25 rgb2sfnorm_25 \
class_places_25 room_layout_25"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/encoder_arch_26/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise_25 rgb2sfnorm_25 \
class_places_25 room_layout_25"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/encoder_arch_26/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done
###################
# RESNET101 Task  #
###################
SRC_TASKS="denoise_101 rgb2sfnorm_101 random \
class_places_101 room_layout_101"

TASKS="denoise_101 rgb2sfnorm_101 \
class_places_101 room_layout_101"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/encoder_arch_101/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise_101 rgb2sfnorm_101 \
class_places_101 room_layout_101"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/encoder_arch_101/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

exit 0

###################
# GEN Task        #
###################
SRC_TASKS="denoise_2 rgb2sfnorm_2 random \
class_places_2 room_layout_2"

TASKS="denoise_2 rgb2sfnorm_2 \
class_places_2 room_layout_2"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image_gen.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/generalization/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise_2 rgb2sfnorm_2 \
class_places_2 room_layout_2"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_gen.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/generalization/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

###################
# DEEP Task       #
###################
SRC_TASKS="denoise rgb2sfnorm random \
class_places room_layout"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_no_img_deep.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/deep/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_deep.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/deep/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done


###################
# SHALLOW Task    #
###################
SRC_TASKS="denoise rgb2sfnorm random \
class_places room_layout"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_no_img_shallow.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/shallow/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_shallow.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/shallow/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done



###################
# FAT Task        #
###################
SRC_TASKS="denoise rgb2sfnorm random \
class_places room_layout"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_no_img_fat.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/fat/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_fat.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/fat/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done



###################
# SLIM Task       #
###################
SRC_TASKS="denoise rgb2sfnorm random \
class_places room_layout"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_no_img_slim.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/slim/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

SRC_TASKS="pixels"

TASKS="denoise rgb2sfnorm \
class_places room_layout"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_arch_slim.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/slim/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done



#export HIDDEN_SIZE=8
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#DATA="16k"
#TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0

#parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_gen.py \
    #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image_gen.py \
    #--target-task {} \
    #--hidden-size $HIDDEN_SIZE \
    #--kernel-size $KERNEL_SIZE \
    #--num-layers $NUM_LAYERS \
    #--finetune-decoder False \
    #--retrain-decoder False \
    #--unlock-decoder True \
    #--target-decoder-func $TARGET_DECODER_FUNC \
    #--data-used $DATA \
    #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/generalization/${TARGET_DECODER_FUNC}/${DATA}/ \
    #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS


exit 0

# Initial Transfers
SRC_TASKS="autoencoder colorization denoise edge2d edge3d \
fix_pose impainting jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist \
rgb2sfnorm vanishing_point pixels"

TASKS="autoencoder colorization denoise edge2d edge3d \
fix_pose impainting jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist \
rgb2sfnorm vanishing_point"


# -------------------------------
# -  DIRECT TRANSFER (SPATIAL)  -
# -------------------------------


# ##################################################################
# # TRAIN DECODERS ON DIFFERENT REPS
# ##################################################################
# SRC_TASKS="autoencoder rgb2depth rgb2sfnorm segment2d room_layout reshade random"
# TASKS="autoencoder rgb2depth rgb2sfnorm segment2d room_layout reshade "
# BOOLS="False True"
# export HIDDEN_SIZE=8
# export KERNEL_SIZE=3
# export NUM_LAYERS=3
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_empty.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --retrain-decoder True \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/scratch/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done


# ##################################################################
# # TRAIN PRETRAINED DECODERS ON DIFFERENT REPS
# ##################################################################
# SRC_TASKS="autoencoder rgb2sfnorm segment2d reshade random"
# TASKS="autoencoder rgb2sfnorm segment2d reshade"
# export HIDDEN_SIZE=8
# export KERNEL_SIZE=3
# export NUM_LAYERS=3
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_empty.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --retrain-decoder True \
#         --finetune-decoder True \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/scratch/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done




# ##################################################################
# # TRAIN ARCHITECTURES FROM SCRATCH
# ##################################################################
# SRC_TASKS="autoencoder rgb2depth rgb2sfnorm segment2d reshade room_layout pixels random"
# TASKS="autoencoder rgb2depth rgb2sfnorm room_layout segment2d reshade"
# export FINETUNE=False
# export UNLOCK=True
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --finetune-decoder $FINETUNE \
#         --unlock-decoder $UNLOCK \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l2.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l2/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l4.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l4/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done
# for i in $SRC_TASKS;
# do
#     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#         --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l8.py \
#         --input-task $i --target-task {} \
#         --hidden-size $HIDDEN_SIZE \
#         --kernel-size $KERNEL_SIZE \
#         --num-layers $NUM_LAYERS \
#         --unlock-decoder $UNLOCK \
#         --finetune-decoder $FINETUNE \
#         --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l8/ \
#         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# done

# ##################################################################
# # SEARCH OVER TRANSFER ARCHITECTURES
# ##################################################################
# SRC_TASKS="autoencoder rgb2depth segment2d room_layout random"
# TASKS="autoencoder rgb2depth segment2d room_layout"
# BOOLS="False True"
# for FINETUNE in $BOOLS;
# do
#     export HIDDEN_SIZE=8
#     export KERNEL_SIZE=3
#     export NUM_LAYERS=3
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l1l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l1l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l3l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l3l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --finetune-decoder $FINETUNE \
#             --retrain-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_l6l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/l6l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l1l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l1l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l3l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l3l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l2.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l2/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l4.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l4/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done
#     for i in $SRC_TASKS;
#     do
#         parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
#             --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l8.py \
#             --input-task $i --target-task {} \
#             --hidden-size $HIDDEN_SIZE \
#             --kernel-size $KERNEL_SIZE \
#             --num-layers $NUM_LAYERS \
#             --retrain-decoder $FINETUNE \
#             --finetune-decoder $FINETUNE \
#             --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_l6l8/ \
#             | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#     done

# done










#SRC_TASKS="pixels random segment2d segment25d rgb2sfnorm reshade vanishing_point_well_defined"
#TASKS="rgb2sfnorm segment2d reshade segment25d vanishing_point_well_defined"
SRC_TASKS="pixels"

TASKS="autoencoder colorization curvature denoise edge2d edge3d \
ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000 class_places"

#BOOLS="False True"
TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
# DATA_USED="1h 5h 1k 2k 4k 8k 16k 23k 30k"
DATA_USED="1k 16k"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
#for TARGET_DECODER_FUNC in $TARGET_DECODER_FUNCS;
#do
    # L2l2L2
#    for i in $SRC_TASKS;
    #do
        #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l2l2l0.py \
            #--input-task $i --target-task {} \
            #--hidden-size $HIDDEN_SIZE \
            #--kernel-size $KERNEL_SIZE \
            #--num-layers $NUM_LAYERS \
            #--finetune-decoder False \
            #--retrain-decoder False \
            #--unlock-decoder True \
            #--target-decoder-func decoder_tiny_transfer_6 \
            #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/decoder_tiny_transfer_6/dilated_l2l2l0/ \
            #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    #done

    # Data split 
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNCS \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_taskonomy_beta1/${TARGET_DECODER_FUNCS}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

######################
# External Baselines #
######################

SRC_TASKS="alex lsm cmu pose5 pose6 poseMatch"

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000 class_places"

#BOOLS="False True"
TARGET_DECODER_FUNCS="DO_NOT_REPLACE_TARGET_DECODER"
# DATA_USED="1h 5h 1k 2k 4k 8k 16k 23k 30k"
DATA_USED="16k"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3

for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image_external.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNCS \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/${TARGET_DECODER_FUNCS}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

echo "DONE Generating config files..."
    # L6L8
#    for i in $SRC_TASKS;
    #do
        #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_l6l8.py \
            #--input-task $i --target-task {} \
            #--hidden-size $HIDDEN_SIZE \
            #--kernel-size $KERNEL_SIZE \
            #--num-layers $NUM_LAYERS \
            #--finetune-decoder False \
            #--retrain-decoder False \
            #--unlock-decoder True \
            #--target-decoder-func $TARGET_DECODER_FUNC \
            #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/${TARGET_DECODER_FUNC}/dilated_l6l8/ \
            #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    #done
#done










# 3. *Run huge decoder for following combination, for sufficient statistics: * (Total 112 experiments)
# Data: 1k, 2k, 16k, 30k
SRC_TASKS="autoencoder colorization curvature denoise edge2d edge3d \
ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined pixels random \
segmentsemantic_rb class_selected class_1000 class_places"

TASKS="autoencoder colorization curvature denoise edge2d edge3d \
ego_motion fix_pose impainting_whole jigsaw keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000 class_places"
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="1k 16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
## mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
## L2l2L0
for i in $SRC_TASKS;
do
    for DATA in $DATA_USED;
    do
        parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
            --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image.py \
            --input-task $i --target-task {} \
            --hidden-size $HIDDEN_SIZE \
            --kernel-size $KERNEL_SIZE \
            --num-layers $NUM_LAYERS \
            --finetune-decoder False \
            --retrain-decoder False \
            --unlock-decoder True \
            --target-decoder-func $TARGET_DECODER_FUNC \
            --data-used $DATA \
            --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/${TARGET_DECODER_FUNC}/${DATA}/ \
            | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
    done
done

# 3. *Run huge decoder for following combination, for sufficient statistics: * (Total 112 experiments)
# Data: 1k, 2k, 16k, 30k


################################
# Second Order Task Config Gen #
################################

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_second_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA="2k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_second_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA="1k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_second_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_second_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --random \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

###############################
# Third Order Task Config Gen #
###############################

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_third_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA="1k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_third_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

################################
# Fourth Order Task Config Gen #
################################

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_fourth_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA="1k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_fourth_order.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

################################
# Higher Order Task Config Gen #
################################

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_1000 class_places"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3

# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

for number in {5..25}
do
    DATA="16k"
    TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
    parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
        --target-task {} --input-task $number\
        --hidden-size $HIDDEN_SIZE \
        --kernel-size $KERNEL_SIZE \
        --num-layers $NUM_LAYERS \
        --finetune-decoder False \
        --retrain-decoder False \
        --unlock-decoder True \
        --target-decoder-func $TARGET_DECODER_FUNC \
        --data-used $DATA \
        --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/high_order/${TARGET_DECODER_FUNC}/${DATA}/ \
        | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

    DATA="1k"
    TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
    # mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
    # L2l2L0

    parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
        --target-task {} --input-task $number\
        --hidden-size $HIDDEN_SIZE \
        --kernel-size $KERNEL_SIZE \
        --num-layers $NUM_LAYERS \
        --finetune-decoder False \
        --retrain-decoder False \
        --unlock-decoder True \
        --target-decoder-func $TARGET_DECODER_FUNC \
        --data-used $DATA \
        --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/high_order/${TARGET_DECODER_FUNC}/${DATA}/ \
        | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
done



###################
# Places Affected #
###################
TASKS="non_fixated_pose segmentsemantic_rb class_places"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_places_as_src.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA="1k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# mkdir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy
# L2l2L0

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs_places_as_src.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/second_order/${TARGET_DECODER_FUNC}/${DATA}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS



##############################
# Full Order Task Config Gen #
##############################

TASKS="autoencoder curvature denoise edge2d edge3d \
ego_motion fix_pose keypoint2d keypoint3d \
non_fixated_pose point_match reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point_well_defined \
segmentsemantic_rb class_selected class_1000"

export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
INPUT_TASK="FULL"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# L2l2L0
parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order_image/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

INPUT_TASK="FULL_select"
parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order_selected/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order_selected_image/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

DATA_USED="1k"
INPUT_TASK="FULL"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
# L2l2L0
parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order_no_image.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS

parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_second_order.py \
    --target-task {} \
    --input-task $INPUT_TASK \
    --hidden-size $HIDDEN_SIZE \
    --kernel-size $KERNEL_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/full_order_image/${TARGET_DECODER_FUNC}/${DATA_USED}/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
# Baseline transfer net
#export HIDDEN_SIZE=8
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_regular.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_regular/ \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

#export HIDDEN_SIZE=8
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_dilated_shallow.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/dilated_shallow/ \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

#export HIDDEN_SIZE=8
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_regular.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/regular/ \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

#export HIDDEN_SIZE=8
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_pix_stream_shallow.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #--dest-dir /home/ubuntu/task-taxonomy-331b/experiments/transfers/pix_stream/shallow/ \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done
## Full-strength transfer net
#export HIDDEN_SIZE=10
#export KERNEL_SIZE=5
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

## Shallow nets with weak convs
#export HIDDEN_SIZE=2
#export KERNEL_SIZE=1
#export NUM_LAYERS=2
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

## Shallow nets with normal convs
#export HIDDEN_SIZE=9
#export KERNEL_SIZE=3
#export NUM_LAYERS=2
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

## Shallow net with few channels and normal convs
#export HIDDEN_SIZE=4
#export KERNEL_SIZE=3
#export NUM_LAYERS=2
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done

## Normal layers with few channels and small convs
#export HIDDEN_SIZE=5
#export KERNEL_SIZE=1
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done




## Unlocked transfer
#SRC_TASKS="autoencoder denoise segment2d rgb2depth"
#TASKS="denoise segment2d rgb2depth"
#export HIDDEN_SIZE=12
#export KERNEL_SIZE=3
#export NUM_LAYERS=3
#for i in $SRC_TASKS;
#do
    #parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
        #--prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_conv.py \
        #--input-task $i --target-task {} \
        #--hidden-size $HIDDEN_SIZE \
        #--kernel-size $KERNEL_SIZE \
        #--num-layers $NUM_LAYERS \
        #--finetune-decoder True \
        #| tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
#done


## # ---------------------------------
## # -  DIRECT TRANSFER (FLATTENED)  -
## # ---------------------------------
## # SRC_TASKS=$TASKS
## export HIDDEN_SIZE=1024
## for i in $SRC_TASKS;
## do
##     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
##         --input-task $i --target-task {} --hidden-size $HIDDEN_SIZE \
##         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
## done

## echo -e "\tHS 256..."
## export HIDDEN_SIZE=256
## for i in $TASKS;
## do
##     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
##         --input-task $i --target-task {} --hidden-size $HIDDEN_SIZE \
##         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
## done

## echo -e "\tHS 64..."
## export HIDDEN_SIZE=64
## for i in $TASKS;
## do
##     parallel --no-notice "python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
##         --input-task $i --target-task {} --hidden-size $HIDDEN_SIZE \
##         | tail -n 1 >> $TRANSFER_COMMAND_FILE;" ::: $TASKS
## done


## # ---------------------------------
## # -  FUNNEL TRANSFER (FLATTENED)  -
## # ---------------------------------
## # Funnel Transfers
## echo -e "\tFunnel transfer..."
## CLUSTERS="2d 3d conv all"
## DST_TASKS="denoise keypoint2d rgb2sfnorm"
## export HIDDEN_SIZE=1024
## for i in $CLUSTERS;
## do
##     for intermediary in $DST_TASKS;
##     do
##         for dst in $DST_TASKS;
##         do
##             python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
##                 --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_funnel.py \
##                 --input-cluster $i \
##                 --intermediary-task $intermediary \
##                 --target-task $dst \
##                 --hidden-size $HIDDEN_SIZE \
##                 | tail -n 1 >> $TRANSFER_COMMAND_FILE;
##         done
##     done
## done

## # ----------------------------------------------------
## # -  SAMPLED STAGED TRANSITIVE TRANSFER (FLATTENED)  -
## # ----------------------------------------------------
## export HIDDEN_SIZE=1024
## # Now generate transfers
## echo -e "\tTransitive transfers..."
## while IFS=' ' read -ra ADDR; do
##     src="${ADDR[0]}";
##     intermediate="${ADDR[1]}";
##     dst="${ADDR[2]}";
##     python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
##     --preinput-task $src --input-task $intermediate --target-task $dst --hidden-size $HIDDEN_SIZE \
##     | tail -n 1 >> $TRANSFER_COMMAND_FILE
## done < /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt

# ------------------------------------------------------
# -  SAMPLED UNLOCKED TRANSITIVE TRANSFER (FLATTENED)  -
# ------------------------------------------------------
echo -e "\t Unlocked transitive transfers..."
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="16k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
while IFS=' ' read -ra ADDR; do
    src="${ADDR[0]}";
    intermediate="${ADDR[1]}";
    dst="${ADDR[2]}";
    # echo $src $intermediate $dst
    python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --preinput-task $src --intermediary-task $intermediate --target-task $dst \
    --hidden-size $HIDDEN_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image_trans.py \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/${TARGET_DECODER_FUNC}/16ktransitive/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE

done < /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples.txt


echo -e "\t Unlocked transitive transfers..."
export HIDDEN_SIZE=8
export KERNEL_SIZE=3
export NUM_LAYERS=3
DATA_USED="1k"
TARGET_DECODER_FUNC=DO_NOT_REPLACE_TARGET_DECODER
while IFS=' ' read -ra ADDR; do
    src="${ADDR[0]}";
    intermediate="${ADDR[1]}";
    dst="${ADDR[2]}";
    # echo $src $intermediate $dst
    python /home/ubuntu/task-taxonomy-331b/tools/script/generate_transfer_configs.py \
    --preinput-task $src --intermediary-task $intermediate --target-task $dst \
    --hidden-size $HIDDEN_SIZE \
    --num-layers $NUM_LAYERS \
    --finetune-decoder False \
    --retrain-decoder False \
    --unlock-decoder True \
    --target-decoder-func $TARGET_DECODER_FUNC \
    --data-used $DATA_USED \
    --prototype_path /home/ubuntu/task-taxonomy-331b/assets/transfers/config_prototype_data_check_no_image_trans.py \
    --dest-dir /home/ubuntu/task-taxonomy-331b/experiments/rep_only_taskonomy/${TARGET_DECODER_FUNC}/1ktransitive/ \
    | tail -n 1 >> $TRANSFER_COMMAND_FILE

done < /home/ubuntu/task-taxonomy-331b/assets/transitivity/selected_samples_1k.txt

echo "Commands in:"
echo "Commands in:"
echo $TRANSFER_COMMAND_FILE
