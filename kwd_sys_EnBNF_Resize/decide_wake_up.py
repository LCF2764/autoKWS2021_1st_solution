import os
import sys
from kws import MyKWS
from tqdm import tqdm

def get_utt2ark_dict(enro_dir, test_dir):
    enro_scp = os.path.join(enro_dir, 'feats.scp')
    test_scp = os.path.join(test_dir, 'feats.scp')

    enro_bnf_ark_dir = {}
    for line in open(enro_scp, 'r').readlines():
        utt, ark = line.strip().split()
        enro_bnf_ark_dir[utt] = ark

    test_bnf_ark_dir = {}
    for line in open(test_scp, 'r').readlines():
        utt, ark = line.strip().split()
        test_bnf_ark_dir[utt] = ark
    return enro_bnf_ark_dir, test_bnf_ark_dir

def predicts(model_path, test_list, enro_dict, test_dict, output_file, threshold_value):
    test_result_dict = {}
    my_kws = MyKWS(model_path)
    for test_utt in tqdm(test_dict):
        test_result_dict[test_utt] = []
        for enro_utt in enro_dict:
            test_ark = test_dict[test_utt]
            enro_ark = enro_dict[enro_utt]
            try:
                pred = my_kws.predict(enro_ark, test_ark)
            except:
                pred = 0.5
                print("#"*100+ '     FUCK YOU BUG!!!')
            test_result_dict[test_utt].append(pred)

    wf = open(output_file, 'w')
    for line in open(test_list, 'r').readlines():
        utt = line.strip()
        s_max = max(test_result_dict[utt])
        if s_max >= threshold_value:
            wf.write('{} {} {:.2f}\n'.format(utt, '1', s_max))
        else:
            wf.write('{} {} {:.2f}\n'.format(utt, '0', s_max))
    wf.close()

    # print(open(output_file).readlines())

def main():
    model_path = sys.argv[1]
    test_list = sys.argv[2]
    enro_bnf_dir = sys.argv[3]
    test_bnf_dir = sys.argv[4]
    threshold_value = float(sys.argv[5])
    output_file = sys.argv[6]

    enro_dict, test_dict = get_utt2ark_dict(enro_bnf_dir, test_bnf_dir)
    predicts(model_path, test_list, enro_dict, test_dict, output_file, threshold_value)


if __name__ == "__main__":
    main()
