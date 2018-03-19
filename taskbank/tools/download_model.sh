##!/usr/bin/env bash

CURRDIR=$(pwd)
BASEDIR=$(dirname "$0")

TASKS="autoencoder \
class_1000 \
class_places \
colorization \
curvature \
denoise \
edge2d \
edge3d \
inpainting_whole \
jigsaw \
keypoint2d \
keypoint3d \
reshade \
rgb2depth \
rgb2mist \
rgb2sfnorm \
room_layout \
segment25d \
segment2d \
segmentsemantic \
vanishing_point"
mkdir -p "$CURRDIR/$BASEDIR/../temp"

SUBFIX="data-00000-of-00001 meta index"

for t in $TASKS; do
    mkdir -p "$CURRDIR/$BASEDIR/../temp/${t}"
    for s in $SUBFIX; do
        echo "Downloading ${t}'s model.${s}"
        wget "https://s3-us-west-2.amazonaws.com/taskonomy-unpacked-oregon/\
model_log_final/${t}/logs/model.permanent-ckpt.${s}" -P $CURRDIR/$BASEDIR/../temp/${t}
    done 
done
