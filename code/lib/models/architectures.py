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
from models.perceptual_transfer import TransferNet
from models.chained_transfer import ChainedTransferNet
from models.constant_predictor import ConstantPredictorSegmentation, ConstantPredictorPose, ConstantPredictorL2
from models.resnet_ff import ResNet_FF
from models.FCRN_depth import FCRN_depth

ALLOWABLE_TYPES = [ 
    BasicFF,
    CycleSiamese,
    EncoderDecoderWithCGAN, 
    EncoderDecoder,
    EDSoftmax, 
    EDSoftmaxRegenCGAN,
    ResNet_FF,
    SegmentationEncoderDecoder, 
    SemSegED,
    Siamese, 
    ConstantPredictorSegmentation,
    ConstantPredictorPose,
    ConstantPredictorL2,
    TransferNet, 
    ChainedTransferNet,
    FCRN_depth,
    'empty' ]
