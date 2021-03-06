# -*- encoding: utf-8 -*-
import os
import sys
# from kaldi_python_io import ScriptReader
import kaldiio
import pdb
import numpy as np


def read_data_dir(input_result):
    utt2result_dict = {}
    h_result = open(input_result, 'r')
    for line_str in h_result:
        if line_str.find("\t") == -1:
            utt, result = line_str.strip().split(" ", 1)
        else:
            utt, result = line_str.strip().split("\t", 1)
        utt2result_dict[utt] = result
    h_result.close()

    return utt2result_dict

def re_decide(enroll_xvector_scp, test_xvector_scp, threshold_value):
    enroll_xvector_dict = {}
    for line in open(enroll_xvector_scp).readlines():
        utt, ark = line.strip().split()
        enroll_xvector = kaldiio.load_mat(ark)
        enroll_xvector_dict[utt] = enroll_xvector

    # enroll_xvector_scp_reader = ScriptReader(enroll_xvector_scp)
    # for utt, value in enroll_xvector_scp_reader:
    #     enroll_xvector = value
    
    re_decide_dict = {}
    # test_xvector_scp_reader = ScriptReader(test_xvector_scp)
    # for utt, test_xvector in test_xvector_scp_reader:
    for line in open(test_xvector_scp).readlines():
        utt, ark = line.strip().split()
        test_xvector = kaldiio.load_mat(ark)

        # 遍历enroll_xvector_dict，求均值
        scores_list = []
        for enroll_utt in enroll_xvector_dict:
            cur_enroll_xvector = enroll_xvector_dict[enroll_utt]
            # cos distance
            score = np.dot(cur_enroll_xvector,test_xvector)/(np.linalg.norm(cur_enroll_xvector)*np.linalg.norm(test_xvector))
            scores_list.append(score)
        # 求均值
        dist = sum(scores_list)/len(scores_list)
        # print(dist)
        if dist >= threshold_value:
            re_decide_dict[utt] = 1
        else:
            re_decide_dict[utt] = 0
    return re_decide_dict

def output(utt2result_dict, re_decide_dict, output_result):
    h_output = open(output_result, 'w')
    for utt, result in utt2result_dict.items():
        if utt in re_decide_dict:
            result = str(re_decide_dict[utt])
        h_output.write("%s %s\n" % (utt, result))
    h_output.close()
        

def main():
    enroll_xvector_scp = sys.argv[1]
    test_xvector_scp = sys.argv[2]
    threshold_value = float(sys.argv[3])
    input_result = sys.argv[4]
    output_result = sys.argv[5]

    utt2result_dict = read_data_dir(input_result)
    re_decide_dict = re_decide(enroll_xvector_scp, test_xvector_scp, threshold_value)
    output(utt2result_dict, re_decide_dict, output_result)


if __name__ == "__main__":
    main()