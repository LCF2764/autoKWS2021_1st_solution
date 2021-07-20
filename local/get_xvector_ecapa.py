# -*- encoding: utf-8 -*-
import argparse
from kaldiio import WriteHelper
import soundfile as sf
import sys, os, torch
import numpy as np 
from tqdm import tqdm
from webrtc_vad import remove_sil_and_save

sys.path.append('speechbrain')
from speechbrain.pretrained import SpeakerRecognition

parser = argparse.ArgumentParser(description='get_xvector_ecapa')
parser.add_argument('--data_dir',   type=str, default='workdir/P0001/data/enroll',    help='path of kaldi-format data dir')
parser.add_argument('--output_dir', type=str, default='workdir/P0001/xvector/enroll', help='path of kaldi-format xvector dir')
parser.add_argument('--ecapa_source', type=str, default='speechbrain/pretrained_models/spkrec-ecapa-voxceleb', help='path of kaldi-format xvector dir')
parser.add_argument('--concat_and_vad', type=bool, default=False, help='concat all the enroll wav and remove sil.')

args = parser.parse_args()

## ini ecapa model
verification = SpeakerRecognition.from_hparams(source=args.ecapa_source, savedir=args.ecapa_source)

def concat_save_vad():
    """
        1.concat all enroll wav file and save it to args.output_dir/all.wav;
        2.compute vad and remove sil;
        3.save the no-sil file to args.output_dir/all_nosil.wav
    """
    ## get wav.scp
    wav_scp_fi = os.path.join(args.data_dir, 'wav.scp')
    ## concat and save
    all_audio = np.array([])
    for line in open(wav_scp_fi, 'r').readlines():
        utt, wav = line.strip().split()
        signal, sr = sf.read(wav)
        all_audio = np.hstack([all_audio, signal])
    sf.write(os.path.join(args.output_dir,'all.wav'), all_audio, samplerate=16000)
    ## vad
    in_wav = os.path.join(args.output_dir,'all.wav')
    out_wav = os.path.join(args.output_dir,'all_nosil.wav')
    remove_sil_and_save(in_wav, out_wav, 3) # the output file will save to $out_wav
    return out_wav


def extract_embedding(args):
    ## get wav.scp
    wav_scp_fi = os.path.join(args.data_dir, 'wav.scp')

    ## extract xvector and write to ark file
    with WriteHelper('ark,scp:{d}/xvector_ecapa.ark,{d}/xvector_ecapa.scp'.format(d=args.output_dir)) as ark_writer:
        ## 如果不拼接注册集和删除静音：
        if not args.concat_and_vad:
            for line in tqdm(open(wav_scp_fi).readlines()):
                utt, wav_fi = line.strip().split()
                
                # signal, fs = torchaudio.load(wav_fi)
                signal, fs = sf.read(wav_fi, dtype="float32")
                signal = torch.from_numpy(signal)

                embedding = verification.encode_batch(signal)

                ark_writer(utt, embedding.squeeze().cpu().numpy())
        else:
            utt = args.data_dir.split('/')[-3]
            wav_fi = os.path.join(args.output_dir,'all_nosil.wav')
            signal, fs = sf.read(wav_fi, dtype="float32")
            signal = torch.from_numpy(signal)
            embedding = verification.encode_batch(signal)
            ark_writer(utt, embedding.squeeze().cpu().numpy())

if args.concat_and_vad:
    concat_save_vad()
extract_embedding(args)


