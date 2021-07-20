from __future__ import print_function

import argparse
import os, yaml
import numpy as np 
import torch
import torchaudio
import kaldiio
import glob
import torchaudio.compliance.kaldi as kaldi
import matplotlib.pyplot as plt
from tqdm import tqdm
from kaldiio import WriteHelper
from scipy.spatial.distance import cdist
from wenet.transformer.bnf_model import init_bnf_model
from wenet.utils.checkpoint import load_checkpoint

parser = argparse.ArgumentParser(description='recognize with your model')
parser.add_argument('--config',     type=str,   default='/home/lcf/speech/bnf_cnn_qbe-std/wenet/20210315_unified_transformer_exp/train.yaml', help='config file')
# parser.add_argument('--gpu',        type=int,   default=0,     help='gpu id for this rank, -1 for cpu')
parser.add_argument('--checkpoint', type=str,   default='encoder.pt', help='checkpoint model')
parser.add_argument('--query_wav',  type=str,   default='/T001E001.wav', help='checkpoint model')
parser.add_argument('--test_wav',   type=str,   default='/T001E002.wav', help='checkpoint model')
parser.add_argument('--data_dir',   type=str,   default='./', help='checkpoint model')
parser.add_argument('--test_dir',   type=str,   default='/home/lcf/speech/auto_KWS2021/practice', help='checkpoint model')
parser.add_argument('--output_dir', type=str,   default='/home/lcf/speech/bnf_cnn_qbe-std/data/autoKWS2021_fbank_data/bnf', help='checkpoint model')
args = parser.parse_args()

def load_checkpoint(model, path):
    self_state = model.state_dict()
    if torch.cuda.is_available():
        loaded_state = torch.load(path)
    else:
        loaded_state = torch.load(path, map_location='cpu')
    for name, param in loaded_state.items():
        origname = name;
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                print("%s is not in the model."%origname)
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)

def bottolecknet_features_single(model, wav_file):
    waveform, sample_rate = torchaudio.load_wav(wav_file)
    waveform = waveform.float()
    fbank = kaldi.fbank(
                    waveform,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    dither=0.1,
                    energy_floor=0.0,
                    sample_frequency=sample_rate
                    )

    feats_lengths = torch.tensor([fbank.size()[0]])
    fbank = fbank.to(device).unsqueeze(0)
    feats_lengths = feats_lengths.to(device)
    bnf = model.extract_bnf(fbank, feats_lengths).squeeze().cpu().detach().numpy()
    return bnf

def bottolecknet_features_from_wav(model, args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with WriteHelper('ark,scp:{d}/encoder_bnf_test.ark,{d}/feats.scp'.format(d=args.output_dir)) as ark_writer:
        for wav in tqdm(glob.glob(args.test_dir + '/P*/*/*.wav')):
            utt = wav.split('/')[-1].split('.')[0]
            if utt[5] == 'E':
                utt = 'enroll_'+utt
            else:
                utt = 'test_'+utt

            waveform, sample_rate = torchaudio.load_wav(wav)
            waveform = waveform.float()
            fbank = kaldi.fbank(
                            waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            dither=0.1,
                            energy_floor=0.0,
                            sample_frequency=sample_rate
                            )

            feats_lengths = torch.tensor([fbank.size()[0]])
            fbank = fbank.to(device).unsqueeze(0)
            feats_lengths = feats_lengths.to(device)
            bnf = model.extract_bnf(fbank, feats_lengths).squeeze().cpu().detach().numpy()
            ark_writer(utt, bnf)

def bottolecknet_features_from_fbank(model, args):
    feat_scp = os.path.join(args.data_dir, 'feats.scp')
    output_dir = args.output_dir
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with WriteHelper('ark,scp:{d}/encoder_bnf.ark,{d}/feats.scp'.format(d=args.output_dir)) as ark_writer:
        for line in tqdm(open(feat_scp, 'r').readlines()):
            utt, ark = line.strip().split()
            fbank = kaldiio.load_mat(ark)
            fbank_len = fbank.shape[0]
            a = 0
            while fbank_len < 11:
                a += 1
                fbank = np.vstack([fbank, fbank])
                fbank_len = fbank.shape[0]
                if a > 10:
                    break
            fbank = torch.from_numpy(fbank).unsqueeze(0).to(device)
            fbank_lengths = torch.tensor([fbank.shape[1]]).to(device)
            bnf = model.extract_bnf(fbank, fbank_lengths).squeeze().cpu().detach().numpy()
            ark_writer(utt, bnf)


if __name__ == '__main__':

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_bnf_model(configs)
    load_checkpoint(model, args.checkpoint)
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    bottolecknet_features_from_fbank(model, args)
    # bottolecknet_features_from_wav(model, args)







    # query = bottolecknet_features(model, args.query_wav)
    # test = bottolecknet_features(model, args.test_wav)



    # qt_dists    = cdist(query, test, 'seuclidean', V = None)
    # qt_dists    = -1 + 2 * ((qt_dists - qt_dists.min())/(qt_dists.max() - qt_dists.min()))
    # plt.imshow(qt_dists)

    # plt.savefig('T001E001-T001E002.png')