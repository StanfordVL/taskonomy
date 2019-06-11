#!/bin/bash

for i in {0..99}
do
    python Debug_viz_for_transfer.py --idx $i --hs 256
done
