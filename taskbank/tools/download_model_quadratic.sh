##!/usr/bin/env bash

CURRDIR=$(pwd)
BASEDIR=$(dirname "$0")

TASKS="ego_motion \
fix_pose \
non_fixated_pose \
point_match"
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
