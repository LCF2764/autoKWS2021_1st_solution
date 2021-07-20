# -*- encoding: utf-8 -*-
# Usage python src/extract_bnf.py sws2013Database_dev_eval/dev_queries/sws2013_dev_001.wav data/sws2013/dev_queries

import sys, os
import numpy as np
from kaldiio import WriteHelper
import glob
from tqdm import tqdm
import argparse

from shennong.audio import Audio
from shennong.features.processor.bottleneck import BottleneckProcessor

parser = argparse.ArgumentParser(description='get_bnf')
parser.add_argument('--in_dir',  type=str, default='data/sws2013Database_dev_eval/Audio',    help='path of kaldi-format data dir')
parser.add_argument('--out_dir', type=str, default='/home/lcf/speech/bnf_cnn_qbe-std/data/sws2013_bnf', help='path of kaldi-format xvector dir')
args = parser.parse_args()

processor = BottleneckProcessor(weights='BabelMulti')

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with WriteHelper('ark,scp:{d}/bnf.ark,{d}/bnf.scp'.format(d=args.out_dir)) as ark_writer:
    for wav_path in tqdm(glob.glob(args.in_dir+'/*.wav')):
        wav_name = wav_path.split('/')[-1]
        audio = Audio.load(wav_path)
        features = processor.process(audio)
        ark_writer(wav_name, features.data)

        # np.save(os.path.join(out_dir, os.path.basename(wav_path) + ".npy"), features.data)
