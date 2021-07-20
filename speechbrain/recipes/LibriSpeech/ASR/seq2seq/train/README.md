# LibriSpeech ASR with seq2seq models (CTC + attention).
This folder contains the scripts to train a seq2seq RNN-based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train.py train/train.yaml

# Results
| Release | hyperparams file | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:|
| 20-05-22 | train_BPE_1000.yaml | 3.08 | https://drive.google.com/drive/folders/1SAndjcThdkO-YQF8kvwPOXlQ6LMT71vt?usp=sharing | 1xV100 32GB |
| 20-05-22 | train_BPE_5000.yaml | 2.89 | https://drive.google.com/drive/folders/1kSwdBT8kDhnmTLzrOPDL77LX_Eq-3Tzl?usp=sharing | 1xV100 32GB |


# Training Time
About N for each epoch with a TESLA V100.
