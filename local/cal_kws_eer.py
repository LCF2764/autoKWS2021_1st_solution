from sklearn import metrics
import numpy as np 
import pandas as pd 


def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return tunedThreshold, eer, fpr, fnr

df = pd.read_csv('sws2013_sample_test-results_2.csv')
labels = list(df['label'])
scores = list(df['pred'])

th, eer, fpr, fnr = tuneThresholdfromScore(scores, labels, [1, 0.1])
print(th)
print(eer)
