import os
import sys
from kws import MyKWS

def get_utt2ark_dict(bnf_dir):
    enro_scp = os.path.join(bnf_dir, 'enroll', 'data', 'feats.scp')
    enro_bnf_ark_dir = {}
    for line in open(enro_scp, 'r').readlines():
        utt, ark = line.strip().split()
        enro_bnf_ark_dir[utt] = ark

    test_scp = os.path.join(bnf_dir, 'test', 'data', 'feats.scp')
    test_bnf_ark_dir = {}
    for line in open(test_scp, 'r').readlines():
        utt, ark = line.strip().split()
        test_bnf_ark_dir[utt] = ark
    return enro_bnf_ark_dir, test_bnf_ark_dir

def predicts(model_path, test_list, enro_dict, test_dict, output_file, threshold_value):
    test_result_dict = {}
    my_kws = MyKWS(model_path)
    for test_utt in test_dict:
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
        if max(test_result_dict[utt]) >= threshold_value:
            wf.write('{} {}\n'.format(utt, '1'))
        else:
            wf.write('{} {}\n'.format(utt, '0'))
    wf.close()


def main():
    model_path = sys.argv[1]
    test_list = sys.argv[2]
    bnf_dir = sys.argv[3]
    threshold_value = float(sys.argv[4])
    output_file = sys.argv[5]

    enro_dict, test_dict = get_utt2ark_dict(bnf_dir)
    predicts(model_path, test_list, enro_dict, test_dict, output_file, threshold_value)


if __name__ == "__main__":
    main()
