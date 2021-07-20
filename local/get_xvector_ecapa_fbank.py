# -*- encoding: utf-8 -*-
import argparse
import kaldiio
from kaldiio import WriteHelper
import soundfile as sf
import sys, os, torch
import numpy as np 
from tqdm import tqdm

sys.path.append('speechbrain')
from speechbrain.pretrained import SpeakerRecognition

parser = argparse.ArgumentParser(description='get_xvector_ecapa')
parser.add_argument('--data_dir_fbank',   type=str, default='workdir/P0001/data_fbank/enroll',    help='path of kaldi-format data dir')
parser.add_argument('--output_dir', type=str, default='workdir/P0001/xvector/enroll', help='path of kaldi-format xvector dir')
parser.add_argument('--ecapa_source', type=str, default='speechbrain/pretrained_models/spkrec-ecapa-voxceleb', help='path of kaldi-format xvector dir')

args = parser.parse_args()

## ini ecapa model
verification = SpeakerRecognition.from_hparams(source=args.ecapa_source, savedir=args.ecapa_source)

## get feats.scp
feats_scp_fi = os.path.join(args.data_dir_fbank, 'feats.scp')

## extract xvector and write to ark file
with WriteHelper('ark,scp:{d}/xvector_ecapa.ark,{d}/xvector_ecapa.scp'.format(d=args.output_dir)) as ark_writer:
    for line in tqdm(open(feats_scp_fi).readlines()):
        utt, ark = line.strip().split()

        fbank = kaldiio.load_mat(ark)        
        fbank = torch.from_numpy(fbank).unsqueeze(0)

        embedding = verification.encode_batch_from_fbank(fbank)

        ark_writer(utt, embedding.squeeze().cpu().numpy())



