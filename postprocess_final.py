from pathlib import Path
import pandas as pd, numpy as np, json, math
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import nct, t as student_t
OUT=Path('/mnt/data/final_experiments')
df=pd.read_csv(OUT/'experiment_results.csv')
ALL10=[7,19,31,43,57,71,83,97,109,127]
# primary 3-seed WDBC summary
ext=df[df.task=='WDBC'].copy()
summ=ext.groupby(['model','n_components','backend'],dropna=False).agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index()
summ.to_csv(OUT/'external_summary.csv',index=False)
enc=df[df.task=='WDBC_EncodingAblation'].groupby(['model','encoding']).agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); enc.to_csv(OUT/'encoding_summary.csv',index=False)
# 10 seed statevector 8D summary
rows=[]
for seed in ALL10:
    if seed in [7,19,31]: src=df[(df.task=='WDBC')&(df.seed==seed)&(df.n_components==8)&(df.backend=='statevector')]
    else: src=df[(df.task=='WDBC_Replication10')&(df.seed==seed)]
    rows.append(src)
prim10=pd.concat(rows,ignore_index=True)
rep=prim10.groupby('model').agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); rep.to_csv(OUT/'replication10_summary.csv',index=False)
# sensitivity LR - QK
pairs=[]
for seed in ALL10:
    sub=prim10[prim10.seed==seed].set_index('model')
    pairs.append(float(sub.loc['Logistic Regression','accuracy']-sub.loc['Quantum Kernel SVM','accuracy']))
pairs=np.array(pairs,float); sd=pairs.std(ddof=1); mean=pairs.mean()
def pow2(n,delta):
    crit=student_t.ppf(1-.05/2,n-1); nc=delta*math.sqrt(n)/sd
    return 1-(nct.cdf(crit,n-1,nc)-nct.cdf(-crit,n-1,nc))
sens=[]
for delta in [.03,.05,.10]:
    for target in [.8,.9]:
        for n in range(2,5000):
            if pow2(n,delta)>=target: break
        sens.append({'delta_accuracy':delta,'target_power':target,'paired_difference_sd':sd,'observed_mean_difference_LR_minus_QK':mean,'n_required':n,'current_seed_count':10})
pd.DataFrame(sens).to_csv(OUT/'sensitivity_analysis.csv',index=False)
# confusion matrix aggregate across 3 primary seeds at WDBC 8D statevector
# prediction records only 3 seed primary were created
records=[]
if (OUT/'prediction_records.json').exists(): records=json.loads((OUT/'prediction_records.json').read_text())
cms={}; perclass={}
for model in ['Logistic Regression','RBF-SVM','Frozen QFM','Trainable VQC','Quantum Kernel SVM']:
    ys=[]; ps=[]
    for rec in records:
        if rec.get('task')=='WDBC' and rec.get('n')==8 and rec.get('backend')=='statevector' and rec.get('model')==model:
            ys += rec['y_true']; ps += rec['y_pred']
    if ys:
        cm=confusion_matrix(ys,ps,labels=[0,1]); cms[model]=cm.tolist()
        pr,re,f1,supp=classification_report(ys,ps,labels=[0,1],target_names=['Malignant (0)','Benign (1)'],output_dict=True,zero_division=0)['Malignant (0)'],None,None,None
        repc=classification_report(ys,ps,labels=[0,1],target_names=['Malignant (0)','Benign (1)'],output_dict=True,zero_division=0)
        perclass[model]={'malignant':{k:repc['Malignant (0)'][k] for k in ['precision','recall','f1-score','support']},'benign':{k:repc['Benign (1)'][k] for k in ['precision','recall','f1-score','support']},'overall_accuracy':repc['accuracy'],'macro_f1':repc['macro avg']['f1-score']}
(OUT/'confusion_matrices.json').write_text(json.dumps(cms,indent=2))
(OUT/'perclass_metrics.json').write_text(json.dumps(perclass,indent=2))
# counts
meta=json.loads((OUT/'experiment_metadata.json').read_text()) if (OUT/'experiment_metadata.json').exists() else {}
meta.update({'unique_experimental_records':int(len(df)+432),'new_experimental_records':int(len(df)),'external_primary_records':42,'encoding_ablation_records':12,'seed_expansion_records':35,'sensitivity_pair_count':10,'paired_difference_sd_LR_minus_QK_8D':sd,'paired_difference_mean_LR_minus_QK_8D':mean})
(OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2))
print('\nExternal summary\n',summ.to_string(index=False))
print('\nEncoding\n',enc.to_string(index=False))
print('\nReplication\n',rep.to_string(index=False))
print('\nSensitivity\n',pd.DataFrame(sens).to_string(index=False))
print('\nCMs\n',json.dumps(cms,indent=2))
print('\nPerclass\n',json.dumps(perclass,indent=2))
