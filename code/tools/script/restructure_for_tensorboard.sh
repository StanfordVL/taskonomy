# Name: restructure_for_tensorboard.sh
#
# Desc:
#   Creates a new directory structure that's more tensorboard-friendly
#   It restructures the dictionary to be grouped by src/dst/arch through
#   symlinking the original dirs
#
# Note:
#   If a task was restarted early on then Tensorboard may crap out because there
#   is more than one event file. In that case, you should remove the offending 
#   file with the following:
#     sudo find $ROOT_DIR -name '*.tfevents*' -size -100 -delete


SRC_TASKS="autoencoder segment2d rgb2sfnorm reshade pixels random"
DST_TASKS="autoencoder segment2d rgb2sfnorm reshade"

MATCHES=$(mktemp)
ROOT_DIR=/home/ubuntu/s3/experiment_models/pix_stream_transfer_8
find  $ROOT_DIR -type d -name '*logs*' > $MATCHES

for SRC in $SRC_TASKS;
do
    for DST in $DST_TASKS;
    do
        mkdir -p $SRC/$DST
        cd $SRC/$DST
        pwd

        QUERY=${SRC}/${DST}/logs
        QUERY_MATCHES=$(mktemp)

        cat $MATCHES | grep $QUERY > $QUERY_MATCHES
        while read f; do
            echo $f
            local_file="${f/$ROOT_DIR/.}"

            mkdir -p $local_file
            ln -s "$f" $local_file
        done < $QUERY_MATCHES
        cd -
    done
done

# QUERY=rt_ft
# QUERY_MATCHES=$(mktemp)
# cat $MATCHES | grep $QUERY > $QUERY_MATCHES
# while read f; do
#     # echo $f
#     sudo rm -rf $f
# done < $QUERY_MATCHES