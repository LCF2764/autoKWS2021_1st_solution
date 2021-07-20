# -*- encoding: utf-8 -*-
import argparse
import kaldiio
from kaldiio import WriteHelper
import soundfile as sf
import sys, os, torch
import numpy as np 
from tqdm import tqdm
import torchaudio
from sklearn import metrics

sys.path.append('speechbrain')
from speechbrain.pretrained import SpeakerRecognition

parser = argparse.ArgumentParser(description='get_xvector_ecapa')
parser.add_argument('--trials',   type=str, default='/home/lcf/speech/auto_KWS2021/trials.txt',        help='path of kaldi-format data dir')
parser.add_argument('--output_dir', type=str, default='/home/lcf/speech/auto_KWS2021/embedding_ecapa', help='path of kaldi-format xvector dir')
parser.add_argument('--ecapa_source', type=str, default='/home/lcf/speech/auto_KWS2021/autokws2021/code_submission/ECAPA_model/spkrec-ecapa-voxceleb', help='path of kaldi-format xvector dir')

args = parser.parse_args()

def extract_embeddings(args):
    ## ini ecapa model
    verification = SpeakerRecognition.from_hparams(source=args.ecapa_source, savedir=args.ecapa_source)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## 读取wav文件并去重
    wav_list = []
    for line in open(args.trials, 'r').readlines():
        gt, utt1, utt2 = line.strip().split()
        wav_list.append(utt1)
        wav_list.append(utt2)
    wav_list = list(set(wav_list))

    ## extract xvector and write to ark file
    with WriteHelper('ark,scp:{d}/xvector_ecapa.ark,{d}/xvector_ecapa.scp'.format(d=args.output_dir)) as ark_writer:
        for wav_fi in tqdm(wav_list):

            utt = wav_fi.split('/')[-1].split('.')[0]

            signal, fs = torchaudio.load(wav_fi)

            embedding = verification.encode_batch(signal)

            ark_writer(utt, embedding.squeeze().cpu().numpy())

def cal_scores(args):
    utt2ark_dict = {}
    for line in open(os.path.join(args.output_dir, 'xvector_ecapa.scp'), 'r').readlines():
        utt, ark = line.strip().split()
        utt2ark_dict[utt] = ark

    all_scores = []
    all_labels = []
    wf = open(args.trials.replace('trials', 'trials_predict'), 'w')
    for line in open(args.trials, 'r').readlines():
        # read trial pair
        gt, utt1_path, utt2_path = line.strip().split()
        utt1 = utt1_path.split('/')[-1].split('.')[0]
        utt2 = utt2_path.split('/')[-1].split('.')[0]

        # load embedding
        embed1 = kaldiio.load_mat(utt2ark_dict[utt1])
        embed2 = kaldiio.load_mat(utt2ark_dict[utt2])

        # cal cosine distance
        score = np.dot(embed1,embed2)/(np.linalg.norm(embed1)*np.linalg.norm(embed2))
        wf.write('{} {:.2f}\t{} {}\n'.format(gt, score, utt1_path, utt2_path))
        all_scores.append(score)
        all_labels.append(int(gt))
    return all_scores, all_labels



def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return tunedThreshold, eer, fpr, fnr

def main():
    # extract_embeddings(args)
    sc, lab = cal_scores(args)
    Threshold, eer, fpr, fnr = tuneThresholdfromScore(sc, lab, [1, 0.1]) # Threshold=0.3637113
    print('The EER is: ', eer)
    print('Threshold is: ', Threshold)

if __name__ == '__main__':
    main()
    


