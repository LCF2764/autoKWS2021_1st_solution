#!/usr/bin/env bash
# author:qjshao@npu-aslp.org

# this code that must be submitted
# ./enrollment.sh ../sample_data/practice/P0001/enroll workdir/P0001

enroll_data=$1 #../practice/P0001/enroll
workdir=$2 #workdir/P0001
stage_start=1
stage_end=7
nj=1

data_dir=${workdir}/data/enroll
data_dir_fbank=${workdir}/data_fbank/enroll
mfcc_dir=${workdir}/mfcc/enroll
fbank_dir=${workdir}/fbank/enroll
mfcc_conf=conf/mfcc_30_16k.conf
fbank_conf=conf/fbank_80_16k.conf
bnf_extractor_dir=exp/bnf_extractor
bnf_dir=${workdir}/bnf/enroll
std_dir=${workdir}/std
feat_type="bnf"
distance_type="cosion"
do_mvn=0
vad_fag=1
xvector_extractor_dir=exp/xvector_extractor
xvector_dir=${workdir}/xvector/enroll

submission_dir=$(cd "$(dirname "$0")"; pwd) 
cd $submission_dir
. ./path.sh
. ./cmd.sh


# prepare_data
if [ $stage_start -le 1 -a $stage_end -ge 1 ]; then
    if [ ! -d  ${data_dir} ];then
        mkdir -p ${data_dir}
        # mkdir -p ${data_dir_fbank}
    fi

    python local/prepare_enrollment_for_one_spk.py \
    ${enroll_data} \
    ${data_dir} \
    || exit;

    # cp -r ${data_dir}/* ${data_dir_fbank}

    utils/data/fix_data_dir.sh ${data_dir} || exit 1;
    # utils/data/fix_data_dir.sh ${data_dir_fbank} || exit 1;

    echo "enrollment.sh stage 1 finished " `date`
fi

# make mfcc
if [ $stage_start -le 2 -a $stage_end -ge 2 ]; then
    if [ ! -d ${mfcc_dir} ];then
        mkdir -p ${mfcc_dir}
    fi

    steps/make_mfcc.sh --cmd "$train_cmd" \
    --nj ${nj} \
    --mfcc-config ${mfcc_conf} \
    ${data_dir} \
    ${mfcc_dir}/log \
    ${mfcc_dir} \
    || exit 1;

    steps/compute_cmvn_stats.sh \
    ${data_dir} \
    ${mfcc_dir}/log \
    ${mfcc_dir} \
    || exit 1;

    utils/fix_data_dir.sh ${data_dir} || exit 1;

    echo "enrollment.sh stage 2 finished " `date`
fi

# vad for enrollment
if [ $stage_start -le 3 -a $stage_end -ge 3 ]; then
    steps/compute_vad_decision.sh \
    --nj ${nj} \
    ${data_dir} \
    ${mfcc_dir}/log \
    ${mfcc_dir} \
    || exit 1;

    utils/fix_data_dir.sh ${data_dir} || exit 1;

    python local/std/make_utt2vad.py \
    ${data_dir}/vad.scp \
    ${data_dir}/utt2vad \
    || exit 1;

    echo "enrollment.sh stage 3 finished " `date`
fi

# get_bottleneck_features
if [ $stage_start -le 4 -a $stage_end -ge 4 ]; then
    if [ ! -d ${bnf_dir} ];then
        mkdir -p ${bnf_dir}
    fi

    local/get_bottleneck_features.sh \
    --nj ${nj} \
    --data_dir ${data_dir} \
    --model_dir ${bnf_extractor_dir} \
    --bnf_dir ${bnf_dir} \
    --log_dir ${bnf_dir}/log \
    --bnf_data_dir ${bnf_dir}/data \
    --bnf_htk_dir ${bnf_dir}/htk \
    || exit 1;

    echo "enrollment.sh stage 4 finished " `date`
fi

# prapare the input file of STD_v5 code, which specially made for
# Q by E task. The main functions of STD_v5 are: 1.averaging the enroll templates, 
# 2.caculate the similarity of templates and test utterance feature.
# More details about STD code: https://github.com/jingyonghou/XY_QByE_STD
if [ $stage_start -le 5 -a $stage_end -ge 5 ]; then
    if [ ! -d ${std_dir} ];then
        mkdir -p ${std_dir}
    fi

    cp ${data_dir}/utt2vad ${std_dir}/enroll_list

    echo "enrollment.sh stage 5 finished " `date`
fi

# average the enroll templates
if [ $stage_start -le 6 -a $stage_end -ge 6 ]; then
    ./STD_v5/template_avg_hierarchical \
    ${bnf_dir}/htk/ \
    ${std_dir}/enroll_list \
    "keyword" \
    $feat_type $distance_type $do_mvn $vad_fag \
    ${std_dir}/ \
    || exit 1;

    echo "enrollment.sh stage 6 finished " `date`
fi

# # make FBANK
# if [ $stage_start -le 7 -a $stage_end -ge 7 ]; then
#     if [ ! -d ${fbank_dir} ];then
#         mkdir -p ${fbank_dir}
#     fi

#     steps/make_fbank.sh --cmd "$train_cmd" \
#     --nj ${nj} \
#     --fbank_config ${fbank_conf} \
#     ${data_dir_fbank} \
#     ${fbank_dir}/log \
#     ${fbank_dir} \
#     || exit 1;

#     utils/fix_data_dir.sh ${data_dir_fbank} || exit 1;

#     echo "enrollment.sh make fbank80 finished " `date`
# fi

# extract xvector_ecapa from fbank
# if [ $stage_start -le 7 -a $stage_end -ge 7 ]; then
#     if [ ! -d ${xvector_dir} ];then
#         mkdir -p ${xvector_dir}
#     fi

#     python ./local/get_xvector_ecapa_fbank.py \
#     --data_dir_fbank ${data_dir_fbank} \
#     --output_dir ${xvector_dir} \
#     --ecapa_source ECAPA_model/spkrec-ecapa-voxceleb\
#     || exit 1;
    
#     echo "enrollment.sh stage 7 finished " `date`
# fi

if [ $stage_start -le 7 -a $stage_end -ge 7 ]; then
    if [ ! -d ${xvector_dir} ];then
        mkdir -p ${xvector_dir}
    fi

    python ./local/get_xvector_ecapa.py \
    --data_dir ${data_dir} \
    --output_dir ${xvector_dir} \
    --ecapa_source ECAPA_model/spkrec-ecapa-voxceleb\
    --concat_and_vad false \
    || exit 1;
    
    echo "enrollment.sh stage 7 finished " `date`
fi