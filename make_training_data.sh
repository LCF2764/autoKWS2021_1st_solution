#!/bin/bash

. ./cmd.sh
. ./path_local.sh
set -e
root=/home/lcf/speech/bnf_cnn_qbe-std/data/autoKWS2021_data
autoKWS_train=/home/lcf/speech/bnf_cnn_qbe-std/data/autoKWS2021_train
data_dir=$root/data
exp=$root/exp
mfcc_dir=$root/mfcc
vaddir=$root/mfcc
bnf_dir=$root/bnf
bnf_extractor_dir=exp/bnf_extractor
musan_root=/home/data/Speech_datasets/musan
rirs_root=/home/data/Speech_datasets/RIRS_NOISES
nj=40
mfcc_conf=conf/mfcc_30_16k.conf

stage=3
##############################################
# 提取数据准备
##############################################
if [ $stage -le 0 ]; then
    if [ ! -d  ${data_dir} ];then
        mkdir -p ${data_dir}
    fi

    python local/prepare_train_data.py ${autoKWS_train} ${data_dir} \
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
# data augmentation
##############################################
if [ $stage -le 3 ]; then
    #=========================================
    # rir-generator
    #=========================================
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data_dir/utt2num_frames > $data_dir/reco2dur
    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, $rirs_root/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, $rirs_root/simulated_rirs/mediumroom/rir_list")

    python steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate 16000 \
      $data_dir $root/data_reverb
      
    cp $data_dir/vad.scp $root/data_reverb/
    utils/copy_data_dir.sh --utt-suffix "-reverb" $root/data_reverb $root/data_reverb.new
    rm -rf $root/data_reverb
    mv $root/data_reverb.new $root/data_reverb

    #=========================================
    # Adding addition noise with MUSAN
    #=========================================
    steps/data/make_musan.sh $musan_root $root/musan

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    # for name in speech noise music; do
    #   utils/data/get_utt2dur.sh $root/musan/musan_${name}
    #   mv $root/musan/musan_${name}/utt2dur $root/musan/musan_${name}/reco2dur
    # done

    # Augment with musan_noise
    python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$root/musan/musan_noise" $data_dir $root/data_noise
    # Augment with musan_music
    python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$root/musan/musan_music" $data_dir $root/data_music
    # Augment with musan_speech
    python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$root/musan/musan_speech" $data_dir $root/data_babble
    utils/combine_data.sh $root/data_aug $root/data_reverb $root/data_noise $root/data_music $root/data_babble
    utils/fix_data_dir.sh $root/data_aug # 92000条

    rm -r $root/data_reverb
    rm -r $root/data_noise
    rm -r $root/data_music
    rm -r $root/data_babble

    #=========================================
    # make mfcc
    #=========================================
    steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} --mfcc-config ${mfcc_conf} $root/data_aug ${mfcc_dir}/log ${mfcc_dir}
    
    steps/compute_cmvn_stats.sh $root/data_aug ${mfcc_dir}/log ${mfcc_dir}

    utils/fix_data_dir.sh $root/data_aug 
    # Combine the clean and augmented list.  This is now roughly
    # double the size of the original clean list.
    utils/combine_data.sh $root/data_combined $root/data_aug $data_dir
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

