#!/usr/bin/env bash
# author:qjshao@npu-aslp.org

# if you want to run shell script like this:
# ./a.sh
# you should add the following command.
# or run shell script like:
# bash a.sh
find . |xargs chmod +x

# this code must be submitted
# pip install jieba
##./enrollment.sh ../sample_data/practice/P0001/enroll workdir/P0001

# source activate py36
pip install kaldi_python_io
pip install kaldiio
pip install tqdm
pip install hyperpyyaml
pip install soundfile
pip install huggingface_hub
pip install joblib
pip install torch torchvision torchaudio
pip install webrtcvad
pip install typeguard
# pip install scipy==1.4.1
# conda install -c coml shennong
# pip install torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/