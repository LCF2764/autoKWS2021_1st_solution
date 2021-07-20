#!/usr/bin/env bash
# author:qjshao@npu-aslp.org

# this code that must be submitted
#./predict.sh ../sample_data/practice/P0001/test workdir/P0001 workdir/P0001/predict.txt

test_data=$1 #../practice/P0001/test
workdir=$2   #workdir/P0001
prediction_file=$3 #workdir/P0001/predict.txt

# source activate py36

submission_dir=$(cd "$(dirname "$0")"; pwd)
cd $submission_dir
. ./path.sh
. ./cmd.sh

stage_start=1
stage_end=20
nj=10

data_dir=${workdir}/data/test
data_dir_fbank=${workdir}/data_fbank/test
mfcc_dir=${workdir}/mfcc/test
fbank_dir=${workdir}/fbank/test
mfcc_conf=conf/mfcc_30_16k.conf
fbank_conf=conf/fbank_80_16k.conf
bnf_extractor_dir=exp/bnf_extractor
bnf_dir=${workdir}/bnf/test
bnf_En=${workdir}/bnf_En/test
enro_bnf_En=${workdir}/bnf_En/enroll
test_enro_bnf_dir=${workdir}/bnf
std_dir=${workdir}/std
feat_type="bnf"
distance_type="cosion"
do_mvn=0
vad_fag=0
result_dir=${workdir}/result
threshold_value_1=0.95 #0.63<eer_th>#0.80 #0.80
threshold_value_2=0.3637
xvector_extractor_dir=exp/xvector_extractor
enroll_xvector_dir=${workdir}/xvector/enroll
xvector_dir=${workdir}/xvector/test

# kws_model_path=kwd_sys/models/model_autoKWS2021_ConvNet_noaug_100x100.pt
# kws_model_path=kwd_sys/models/model_AUG_all_100x300-e007.pt   #在th0.9时成绩0.5006
# kws_model_path=kwd_sys/models/model_AUG_all_100x300-e007.pt   #在th0.85时成绩0.5111
# kws_model_path=kwd_sys/models/model_AUG_all_new_100x300-e001.pt #在th0.9时成绩0.5782
kws_model_path=kwd_sys_EnBNF_Resize/models/model_EncoBnf-ConvNet_resize13-e016.pt #在th0.9时成绩 0.6587
# kws_model_path=kwd_sys_EnBNF_RSE34L_Resize/models/ResNetSE34L_G2-model-e002.pt #在th0.9时成绩 

export NPY_MKL_FORCE_INTEL=1

############################################################################
#      stage 1                   准备数据集
############################################################################
if [ $stage_start -le 1 -a $stage_end -ge 1 ]; then
    if [ ! -d  ${data_dir} ];then
        mkdir -p ${data_dir}
    fi
    if [ ! -d  ${data_dir_fbank} ];then
        mkdir -p ${data_dir_fbank}
    fi    
    python local/prepare_test_for_one_spk.py ${test_data} ${data_dir} || exit;
    cp -r ${data_dir}/* ${data_dir_fbank}
    utils/data/fix_data_dir.sh ${data_dir} || exit 1;
    utils/data/fix_data_dir.sh ${data_dir_fbank} || exit 1;
    echo "========== pradict.sh stage 1: Prepare data finished " `date` "=========="
fi

############################################################################
#      stage 2                 提取MFCC，并做VAD
############################################################################
if [ $stage_start -le 2 -a $stage_end -ge 2 ]; then
    if [ ! -d ${mfcc_dir} ];then
        mkdir -p ${mfcc_dir}
    fi
    steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} --mfcc-config ${mfcc_conf} \
                ${data_dir} ${mfcc_dir}/log ${mfcc_dir} || exit 1;
    steps/compute_cmvn_stats.sh ${data_dir} ${mfcc_dir}/log ${mfcc_dir} || exit 1;
    utils/fix_data_dir.sh ${data_dir} || exit 1;
    echo "========== pradict.sh stage 2: Make MFCC finished " `date` "   =========="
fi

############################################################################
#       stage 3                 提取Fbank-80特征
############################################################################
if [ $stage_start -le 3 -a $stage_end -ge 3 ]; then
    if [ ! -d ${fbank_dir} ];then
        mkdir -p ${fbank_dir}
    fi
    steps/make_fbank.sh --write-utt2num-frames true --fbank-config ${fbank_conf} --nj ${nj} --cmd "$train_cmd" \
        ${data_dir_fbank} ${fbank_dir}/log ${fbank_dir} || exit 1;
    utils/fix_data_dir.sh  ${data_dir_fbank} || exit 1;
    echo "========== predict.sh stage 3: Make Fbank80 finished " `date` "=========="
fi

# ############################################################################
# #      stage 4             使用wenet提取bnf特征
# ############################################################################
if [ $stage_start -le 4 -a $stage_end -ge 4 ]; then
    python wenet/extract_bnf.py \
            --config wenet/20210315_unified_transformer_exp/train.yaml \
            --checkpoint wenet/20210315_unified_transformer_exp/encoder.pt \
            --data_dir ${data_dir_fbank} \
            --output_dir ${bnf_En} \
            || exit 1;
    echo "========== predict.sh stage 4: wenet bnf finished " `date` "   =========="
fi

############################################################################
#        stage 5              提取kaldi-bnf特征
############################################################################
if [ $stage_start -le 5 -a $stage_end -ge 5 ]; then
    if [ ! -d ${bnf_dir} ];then
        mkdir -p ${bnf_dir}
    fi
    local/get_bottleneck_features.sh --nj ${nj} \
                --data_dir ${data_dir} \
                --model_dir ${bnf_extractor_dir} \
                --bnf_dir ${bnf_dir} \
                --log_dir ${bnf_dir}/log \
                --bnf_data_dir ${bnf_dir}/data \
                --bnf_htk_dir ${bnf_dir}/htk \
                || exit 1;
    echo "========== predict.sh stage 5: kaldi-bnf finished " `date` "   =========="
fi

############################################################################
#        stage 6    prapare the input file of STD_v5 code
############################################################################
if [ $stage_start -le 6 -a $stage_end -ge 6 ]; then
    if [ ! -d ${std_dir} ];then
        mkdir -p ${std_dir}
    fi
    awk '{print $1}' ${data_dir}/utt2spk > ${std_dir}/test_list
    echo "========== predict.sh stage 6：prapare STD_v5 finished " `date` "=========="
fi

##########################################################################################################
###################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#########################
##########################################################################################################

############################################################################
#        stage 7    使用dtw th0.8 检测关键词: wake_up_result-0 0.62
############################################################################
if [ $stage_start -le 7 -a $stage_end -ge 7 ]; then
    echo keyword_n > ${std_dir}/template_list
    ./STD_v5/dtw_std \
                ${std_dir}/ \
                ${std_dir}/template_list \
                ${bnf_dir}/htk/ \
                ${std_dir}/test_list \
                $feat_type $distance_type $do_mvn \
                ${std_dir}/ || exit 1;
    # decide whether to wake up
    if [ ! -d ${result_dir} ];then
        mkdir -p ${result_dir}
    fi  
    python local/decide_wake_up.py ${std_dir}/test_list ${std_dir}/keyword_n.RESULT \
                0.8 \
                ${result_dir}/wake_up_result-0 \
                || exit 1;
    echo "========== predict.sh stage 7: dtw key-word-detect finished " `date` "  =========="
fi

############################################################################
#   stage 8   使用EnBnf_rResize_ep16_th0.9 th0.9 检测关键词: wake_up_result-1 0.5006
############################################################################
if [ $stage_start -le 8 -a $stage_end -ge 8 ]; then
    python kwd_sys/decide_wake_up.py \
                kwd_sys/models/model_AUG_all_100x300-e007.pt \
                ${std_dir}/test_list \
                ${test_enro_bnf_dir} \
                0.9 \
                ${result_dir}/wake_up_result-1 \
                || exit 1;
    echo "========== predict.sh stage 8: k-bnf-all-Conv_e7 kwd finished " `date` "=========="
fi

############################################################################
#   stage 9  使用kwd_sys_EnBNF_Resize th0.9 检测关键词: wake_up_result-2 0.497
############################################################################
#kwd_sys_EnBNF_Resize/models/model_EncoBnf-ConvNet_resize13-e016.pt
if [ $stage_start -le 9 -a $stage_end -ge 9 ]; then
    python kwd_sys_EnBNF_Resize/decide_wake_up.py \
                kwd_sys_EnBNF_Resize/models/model_EncoBnf-ConvNet_resize13-e016.pt \
                ${std_dir}/test_list \
                ${enro_bnf_En} \
                ${bnf_En} \
                0.9 \
                ${result_dir}/wake_up_result-2 \
                || exit 1;
    echo "========== predict.sh stage 9: kwd_sys_EnBNF_Resize-e16 kwd finished " `date` "=========="
fi

############################################################################
#   stage 10  使用ConvNet_kaldiBNF_Resize-e2 th0.98 检测关键词: wake_up_result-3 
############################################################################
if [ $stage_start -le 10 -a $stage_end -ge 10 ]; then
    python kwd_sys_Resize/decide_wake_up.py \
                kwd_sys_Resize/models/model-ConvNet_kaldiBNF_Resize100x300-e002.pt \
                ${std_dir}/test_list \
                ${test_enro_bnf_dir} \
                0.98 \
                ${result_dir}/wake_up_result-3 \
                || exit 1;
    echo "========== predict.sh stage 10: ConvNet_kaldiBNF_Resize-e2 kwd finished " `date` "=========="
fi

############################################################################
#   stage 11  使用ConvNet_kaldiBNF_Resize-e15 th0.98 检测关键词: wake_up_result-4 
############################################################################
if [ $stage_start -le 10 -a $stage_end -ge 10 ]; then
    python kwd_sys_Resize_e19/decide_wake_up.py \
                kwd_sys_Resize_e19/models/model-e019.pt \
                ${std_dir}/test_list \
                ${test_enro_bnf_dir} \
                0.95 \
                ${result_dir}/wake_up_result-4 \
                || exit 1;
    echo "========== predict.sh stage 11: ConvNet_kaldiBNF_Resize-e15 kwd finished " `date` "=========="
fi


############################################################################
#    stage 7               融合Kws结果, 写入 wake_up_result_ensemble
############################################################################
if [ $stage_start -le 7 -a $stage_end -ge 7 ]; then
    python local/ensemble_wake_up_result.py ${std_dir}/test_list ${result_dir} ${result_dir}/wake_up_result_ensemble || exit 1;
    echo "========== predict.sh stage 9: ensemble wake_up_result finished " `date` "=========="
fi

############################################################################
#    stage 8             prepare sub data dir for wake up data
############################################################################
if [ $stage_start -le 8 -a $stage_end -ge 8 ]; then
    tmpdir=temp_${RANDOM}
    mkdir $tmpdir
    trap 'rm -rf "$tmpdir"' EXIT
    awk '{if ($2==1) print $1}' ${result_dir}/wake_up_result_ensemble > ${tmpdir}/wake_up_list
    utils/data/subset_data_dir.sh --utt-list ${tmpdir}/wake_up_list ${data_dir} ${xvector_dir}/data \
                || exit 1;
    echo "========== predict.sh stage 7: subdata for wake up finished " `date` "=========="
fi

############################################################################
#     stage 9               make ecapa-embedding
############################################################################
if [ $stage_start -le 9 -a $stage_end -ge 9 ]; then
    if [ ! -d ${xvector_dir} ];then
        mkdir -p ${xvector_dir}
    fi
    python ./local/get_xvector_ecapa.py \
                --data_dir ${xvector_dir}/data \
                --output_dir ${xvector_dir} \
                --ecapa_source ECAPA_model/spkrec-ecapa-voxceleb\
                || exit 1;
    echo "========== predict.sh stage 7: ecapa finished " `date` "              =========="
fi

############################################################################
#     stage 10               re-decide wake up 
############################################################################
if [ $stage_start -le 10 -a $stage_end -ge 10 ]; then
    python local/decide_wake_up_again.py \
                ${enroll_xvector_dir}/xvector_ecapa.scp \
                ${xvector_dir}/xvector_ecapa.scp \
                $threshold_value_2 \
                ${result_dir}/wake_up_result_ensemble \
                ${result_dir}/wake_up_result_final \
                || exit 1;
    
    # sed "s/test_\(.*\) /\1.wav /" ${result_dir}/wake_up_result_final |sort > $prediction_file
    sed "s/test_\(.*\) /\1.wav /" ${result_dir}/wake_up_result_final > $prediction_file

    echo "========== predict.sh stage 8: decide_wake_up_again finished" `date` "=========="
fi

