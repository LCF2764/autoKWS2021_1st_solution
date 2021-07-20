#!/bin/bash

. ./cmd.sh
. ./path_local.sh
set -e
root=autoKWS2021_data
autoKWS_test=/home/lcf/speech/bnf_cnn_qbe-std/data/autoKWS2021_test
data_dir=$root/data
exp=$root/exp
mfcc_dir=$root/mfcc
vaddir=$root/mfcc
bnf_dir=$root/bnf
bnf_extractor_dir=exp/bnf_extractor
nj=1
mfcc_conf=conf/mfcc_30_16k.conf

stage=3
##############################################
# 提取数据准备
##############################################
if [ $stage -le 0 ]; then
    if [ ! -d  ${data_dir} ];then
        mkdir -p ${data_dir}
    fi

    python local/prepare_train_data.py ${autoKWS_test} ${data_dir} \
    || exit;

    utils/data/fix_data_dir.sh ${data_dir} \
    || exit 1;

##############################################
# 提取mfcc
##############################################
if [ $stage -le 1 ]; then
    if [ ! -d ${mfcc_dir} ];then
        mkdir -p ${mfcc_dir}
    fi

    steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} --mfcc-config ${mfcc_conf} ${data_dir} ${mfcc_dir}/log ${mfcc_dir} \    
    || exit 1;

    steps/compute_cmvn_stats.sh ${data_dir} ${mfcc_dir}/log ${mfcc_dir} \
    || exit 1;

    utils/fix_data_dir.sh ${data_dir} || exit 1;

    echo "enrollment.sh stage 2 finished " `date`
fi

##############################################
# vad
##############################################
if [ $stage -le 2 ]; then
    steps/compute_vad_decision.sh --nj ${nj} ${data_dir} ${mfcc_dir}/log ${mfcc_dir} \
    || exit 1;

    utils/fix_data_dir.sh ${data_dir} || exit 1;

    python local/std/make_utt2vad.py ${data_dir}/vad.scp ${data_dir}/utt2vad \
    || exit 1;

    echo "enrollment.sh stage 3 finished " `date`
fi

##############################################
# 提取bnf特征
##############################################
if [ $stage -le 4 ]; then
    if [ ! -d ${bnf_dir} ];then
        mkdir -p ${bnf_dir}
    fi

    # 将数据copy到当前docker容器的工作路径下，
    # 然后提取bnf特征
    cd ../
    cp -r $root/data_combined autoKWS2021_data
    cp -r $root/mfcc autoKWS2021_data

    local/get_bottleneck_features.sh \
                                    --nj ${nj} \
                                    --data_dir autoKWS2021_data/data_combined \
                                    --model_dir ${bnf_extractor_dir} \
                                    --bnf_dir autoKWS2021_data/bnf \
                                    --log_dir autoKWS2021_data/bnf/log \
                                    --bnf_data_dir autoKWS2021_data/bnf/data \
                                    --bnf_htk_dir autoKWS2021_data/bnf/htk \
    || exit 1;

    echo "enrollment.sh stage 4 finished " `date`
fi

