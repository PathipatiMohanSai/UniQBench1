from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
OUT=Path('/mnt/data/final_experiments/figures'); OUT.mkdir(exist_ok=True)
df=pd.read_csv('/mnt/data/final_experiments/experiment_results.csv')
# External validation 3-seed 8D statevector, 600dpi
ext=df[(df.task=='WDBC')&(df.n_components==8)&(df.backend=='statevector')].copy()
s=ext.groupby('model').agg(mean=('accuracy','mean'),sd=('accuracy','std')).reindex(['Logistic Regression','RBF-SVM','Quantum Kernel SVM','Frozen QFM','Trainable VQC'])
fig,ax=plt.subplots(figsize=(7.2,3.8),constrained_layout=True)
x=np.arange(len(s)); ax.bar(x,s['mean'],yerr=s['sd'],capsize=3)
ax.set_xticks(x,['Logistic\nRegression','RBF-SVM','Quantum\nKernel SVM','Frozen QFM','Trainable\nVQC'])
ax.set_ylabel('Mean test accuracy'); ax.set_ylim(0.5,1.02); ax.set_title('External validation on WDBC (8-component representation)')
ax.grid(axis='y',linewidth=.4,alpha=.3)
fig.savefig(OUT/'fig6_external_validation.png',dpi=600,bbox_inches='tight'); plt.close(fig)
# Encoding ablation
en=df[df.task=='WDBC_EncodingAblation'].copy(); s=en.groupby(['model','encoding']).agg(mean=('accuracy','mean'),sd=('accuracy','std')).reset_index()
models=['Frozen QFM','Quantum Kernel SVM']; encs=['angle_ry','zz_feature']; labels=['Angle-RY','ZZ feature map']
fig,ax=plt.subplots(figsize=(7.2,3.8),constrained_layout=True); x=np.arange(2); w=.32
for j,m in enumerate(models):
    z=s[s.model==m].set_index('encoding').reindex(encs); ax.bar(x+(j-.5)*w,z['mean'],w,yerr=z['sd'],capsize=3,label=m)
ax.set_xticks(x,labels); ax.set_ylabel('Mean test accuracy'); ax.set_ylim(.55,1.02); ax.set_title('Encoding-choice ablation on WDBC (8 components)'); ax.legend(frameon=False); ax.grid(axis='y',linewidth=.4,alpha=.3)
fig.savefig(OUT/'fig7_encoding_ablation.png',dpi=600,bbox_inches='tight'); plt.close(fig)
# Replication 10 seeds
rep=pd.read_csv('/mnt/data/final_experiments/replication10_summary.csv').set_index('model').reindex(['Logistic Regression','RBF-SVM','Quantum Kernel SVM','Frozen QFM','Trainable VQC'])
fig,ax=plt.subplots(figsize=(7.2,3.8),constrained_layout=True); x=np.arange(len(rep)); ax.bar(x,rep['accuracy_mean'],yerr=rep['accuracy_sd'],capsize=3)
ax.set_xticks(x,['Logistic\nRegression','RBF-SVM','Quantum\nKernel SVM','Frozen QFM','Trainable\nVQC']); ax.set_ylabel('Mean test accuracy'); ax.set_ylim(.5,1.02); ax.set_title('Ten-seed replication on WDBC (8-component statevector)'); ax.grid(axis='y',linewidth=.4,alpha=.3)
fig.savefig(OUT/'fig8_seed_replication.png',dpi=600,bbox_inches='tight'); plt.close(fig)
# Confusion matrices 3-seed aggregated per primary external validation.
cms=json.loads(Path('/mnt/data/final_experiments/confusion_matrices.json').read_text()); per=json.loads(Path('/mnt/data/final_experiments/perclass_metrics.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(8.8,5.4),constrained_layout=True); axs=axs.ravel(); names=list(cms.keys())
for i,name in enumerate(names):
    ax=axs[i]; cm=np.array(cms[name]); im=ax.imshow(cm,aspect='equal');
    for a in range(2):
        for b in range(2): ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=9)
    ax.set_xticks([0,1],['M','B']); ax.set_yticks([0,1],['M','B']); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(name,fontsize=9)
axs[-1].axis('off'); fig.suptitle('Aggregated confusion matrices across three primary WDBC seeds (8D, statevector)',fontsize=11)
fig.savefig(OUT/'fig9_error_analysis.png',dpi=600,bbox_inches='tight'); plt.close(fig)
print('saved', sorted(p.name for p in OUT.glob('fig*.png')))
