''' architectures.py

    Contains high-level model architectures assembled from smaller parts 
'''
from __future__ import absolute_import, division, print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.encoder_decoder import StandardED as EncoderDecoder
from models.encoder_decoder_cgan import EDWithCGAN as EncoderDecoderWithCGAN
from models.encoder_decoder_cgan_softmax import EDWithSoftmaxRegenerationCGAN as EDSoftmaxRegenCGAN
from models.siamese_nets import StandardSiamese as Siamese
from models.cycle_siamese_nets import CycleSiamese as CycleSiamese
from models.basic_feedforward import StandardFeedforward as BasicFF
from models.encoder_decoder_segmentation import SegmentationED as SegmentationEncoderDecoder
from models.encoder_decoder_segmentation_semantic import SemSegED
from models.encoder_decoder_softmax import SoftmaxED as EDSoftmax

ALLOWABLE_TYPES = [ 
    BasicFF,
    CycleSiamese,
    EncoderDecoderWithCGAN, 
    EncoderDecoder,
    EDSoftmax, 
    EDSoftmaxRegenCGAN,
    SegmentationEncoderDecoder, 
    SemSegED,
    Siamese, 
    'empty' ]
