import glob
import sys

test_list_file          = sys.argv[1]
result_dir              = sys.argv[2]
wake_up_result_ensemble = sys.argv[3]

## read predicts to dict
results_dict = {}
for result_fi in glob.glob(result_dir + '/wake_up_result-*'):
    for line in open(result_fi).readlines():
        temp = line.strip().split()
        utt, pred = temp[0], temp[1]
        if utt not in results_dict:
            results_dict[utt] = []
        results_dict[utt].append(pred)

## ensemble
wf = open(wake_up_result_ensemble, 'w')
for line in open(test_list_file).readlines():
    utt = line.strip()
    cur_utt_pred = results_dict[utt]
    ensem_pred = '1' if cur_utt_pred.count('1') >= cur_utt_pred.count('0') else '0'
    wf.write('{} {}\n'.format(utt, ensem_pred))

wf.close()
