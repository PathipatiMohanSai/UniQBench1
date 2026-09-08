from __future__ import annotations
import json, math, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from scipy.stats import nct, t as student_t

OUT = Path('/mnt/data/final_experiments')
OUT.mkdir(exist_ok=True)
RNG_SEEDS = [7,19,31]
EXTRA_SEEDS = [43,57,71,83,97,109,127]
ALL10 = RNG_SEEDS + EXTRA_SEEDS
SHOTS=256
NOISE_P=0.01
DEPTH=2
NCOMP_LIST=[4,8]

# ---------- vectorized simulator ----------
def ry_matrix(angle):
    c=np.cos(angle/2); s=np.sin(angle/2)
    return np.array([[c,-s],[s,c]],dtype=np.complex128)

def rz_matrix(angle):
    return np.array([[np.exp(-1j*angle/2),0],[0,np.exp(1j*angle/2)]],dtype=np.complex128)

def apply_single_batch(states, gate, wire, n):
    a=states.reshape((-1,)+(2,)*n)
    a=np.moveaxis(a, wire+1, 1)
    a=np.einsum('ab,nb...->na...', gate, a)
    a=np.moveaxis(a,1,wire+1)
    return a.reshape((-1,2**n))

def apply_cnot_batch(states, control, target, n):
    dim=2**n
    out=states.copy()
    cbit=1<<(n-1-control); tbit=1<<(n-1-target)
    for idx in range(dim):
        if (idx & cbit) and not (idx & tbit):
            j=idx|tbit
            out[:,idx]=states[:,j]
            out[:,j]=states[:,idx]
    return out

def expectations_batch(states,n):
    probs=np.abs(states)**2
    out=np.empty((states.shape[0],n),dtype=np.float32)
    idx=np.arange(2**n)
    for w in range(n):
        bit=1<<(n-1-w)
        signs=np.where((idx & bit)==0,1.0,-1.0)
        out[:,w]=probs@signs
    return out

def angle_features(X, train_X):
    # training-only scaling to [-pi,pi]
    max_abs=np.maximum(np.max(np.abs(train_X),axis=0,keepdims=True),1e-8)
    return np.clip((X/max_abs)*np.pi,-np.pi,np.pi).astype(np.float32)

def statevector_angles(Xang, params):
    n=params.shape[1]; depth=params.shape[0]
    states=np.zeros((len(Xang),2**n),dtype=np.complex128); states[:,0]=1.0
    for w in range(n):
        a=states.reshape((-1,)+(2,)*n)
        a=np.moveaxis(a,w+1,1)
        c=np.cos(Xang[:,w]/2); s=np.sin(Xang[:,w]/2)
        new=np.empty_like(a)
        new[:,0,...]=c[:,None].reshape(-1,*([1]*(n-1)))*a[:,0,...] - s[:,None].reshape(-1,*([1]*(n-1)))*a[:,1,...]
        new[:,1,...]=s[:,None].reshape(-1,*([1]*(n-1)))*a[:,0,...] + c[:,None].reshape(-1,*([1]*(n-1)))*a[:,1,...]
        a=np.moveaxis(new,1,w+1); states=a.reshape((-1,2**n))
    for d in range(depth):
        states=apply_single_batch(states,rz_matrix(0),0,n) if False else states
        for w in range(n):
            states=apply_single_batch_param(states, params[d,w,0], w, n, 'ry')
            states=apply_single_batch_param(states, params[d,w,1], w, n, 'rz')
        for w in range(n-1): states=apply_cnot_batch(states,w,w+1,n)
    return states

def apply_single_batch_param(states,angle,wire,n,kind):
    a=states.reshape((-1,)+(2,)*n)
    a=np.moveaxis(a,wire+1,1)
    if kind=='ry':
        c=math.cos(angle/2); s=math.sin(angle/2)
        b0=c*a[:,0,...]-s*a[:,1,...]; b1=s*a[:,0,...]+c*a[:,1,...]
    else:
        b0=np.exp(-1j*angle/2)*a[:,0,...]; b1=np.exp(1j*angle/2)*a[:,1,...]
    a=np.stack([b0,b1],axis=1)
    a=np.moveaxis(a,1,wire+1)
    return a.reshape((-1,2**n))

def statevector_zz(Xang, reps=1):
    n=Xang.shape[1]
    states=np.zeros((len(Xang),2**n),dtype=np.complex128); states[:,0]=1.0
    # H gates
    H=np.array([[1,1],[1,-1]],dtype=np.complex128)/math.sqrt(2)
    for w in range(n): states=apply_single_batch(states,H,w,n)
    # ZZ feature map: RZ(x) and ZZ phase x_i*x_j (linear chain)
    for _ in range(reps):
        for w in range(n): states=apply_single_batch_param(states,2*float(np.mean(Xang[:,w])*0),w,n,'rz')
        # use per-sample diagonal phases for accurate data dependence
        for w in range(n):
            # RZ with angle x by sample
            states=apply_rz_sample(states,Xang[:,w],w,n)
        for w in range(n-1): states=apply_zz_sample(states, Xang[:,w], Xang[:,w+1], w,w+1,n)
    return states

def apply_rz_sample(states, angles, wire,n):
    a=states.reshape((-1,)+(2,)*n)
    a=np.moveaxis(a,wire+1,1)
    e0=np.exp(-1j*angles/2).reshape(-1,*([1]*(n-1)))
    e1=np.exp(1j*angles/2).reshape(-1,*([1]*(n-1)))
    a[:,0,...]*=e0; a[:,1,...]*=e1
    a=np.moveaxis(a,1,wire+1)
    return a.reshape((-1,2**n))

def apply_zz_sample(states,a1,a2,w1,w2,n):
    # exp(-i * a1*a2 * Z⊗Z / 2) convention
    a=states.reshape((-1,)+(2,)*n)
    phase=(a1*a2/2)
    idx=np.indices((2,)*n)
    z1=1-2*idx[w1]; z2=1-2*idx[w2]
    diag=np.exp(-1j*phase.reshape((-1,)+(1,)*n)*z1*z2)
    a=a*diag
    return a.reshape((-1,2**n))

def sample_expectations(states, shots, noise_p, seed, depth=2):
    ex=expectations_batch(states, states.ndim-1 if False else int(round(math.log2(states.shape[1]))))
    rng=np.random.default_rng(seed)
    ex2=ex*((1-2*noise_p)**depth)
    p=np.clip((1+ex2)/2,0,1)
    counts=rng.binomial(shots,p)
    return (2*counts/shots-1).astype(np.float32)

def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def fit_head(X,y,seed,epochs=8,lr=.02):
    rng=np.random.default_rng(seed); W=rng.normal(0,.05,size=(X.shape[1],2)).astype(np.float64); b=np.zeros(2)
    for _ in range(epochs):
        order=rng.permutation(len(y))
        for st in range(0,len(order),16):
            idx=order[st:st+16]; xb=X[idx]; yb=y[idx]
            p=softmax(xb@W+b); oh=np.zeros_like(p); oh[np.arange(len(yb)),yb]=1
            g=(p-oh)/len(yb); W-=lr*(xb.T@g); b-=lr*g.sum(axis=0)
    return W,b

def eval_head(X,y,W,b):
    p=softmax(X@W+b); pred=p.argmax(1); return pred,p

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro'))}

# VQC SPSA-lite with minibatch + classical head
# Uses same circuit family and 80 iterations; objective uses 16 sampled training examples per iteration.
def run_vqc(Xtr,ytr,Xv,yv,Xte,yte,n,seed,backend='statevector',encoding='angle_ry',depth=2,iterations=80):
    rng=np.random.default_rng(seed)
    params=rng.normal(0,.5,size=(depth,n,2))
    W=rng.normal(0,.05,size=(n,2)); b=np.zeros(2)
    # cache mini-batches indices; use deterministic permutation each step
    for it in range(iterations):
        idx=rng.choice(len(ytr),size=min(16,len(ytr)),replace=False)
        xb=Xtr[idx]; yb=ytr[idx]
        f_base=simulate_features(xb,params,encoding,backend,seed+10000+it,n,depth)
        p=softmax(f_base@W+b); oh=np.zeros_like(p); oh[np.arange(len(yb)),yb]=1
        g=(p-oh)/len(yb); W-=.05*(f_base.T@g); b-=.05*g.sum(axis=0)
        delta=rng.choice([-1.,1.],size=params.shape)
        c=0.15/(it+1)**0.101
        vp=simulate_features(xb,params+c*delta,encoding,backend,seed+20000+it,n,depth)
        vm=simulate_features(xb,params-c*delta,encoding,backend,seed+30000+it,n,depth)
        lp=-np.mean(np.log(np.clip(softmax(vp@W+b)[np.arange(len(yb)),yb],1e-8,1)))
        lm=-np.mean(np.log(np.clip(softmax(vm@W+b)[np.arange(len(yb)),yb],1e-8,1)))
        grad=(lp-lm)/(2*c)*delta
        eta=.1/(it+1)**0.602
        params-=eta*grad
    trf=simulate_features(Xtr,params,encoding,backend,seed+40000,n,depth)
    vf=simulate_features(Xv,params,encoding,backend,seed+50000,n,depth)
    tef=simulate_features(Xte,params,encoding,backend,seed+60000,n,depth)
    # select threshold/class head with validation, but just fit linear head fixed params
    W,b=fit_head(trf,ytr,seed,epochs=8,lr=.02)
    pv,_=eval_head(vf,yv,W,b); pe,_=eval_head(tef,yte,W,b)
    return pe, {'params':params.tolist(),'backend':backend}

def simulate_features(X,params,encoding,backend,seed,n,depth):
    if encoding=='angle_ry': states=statevector_angles(X,params)
    elif encoding=='zz_feature': states=statevector_zz(X,reps=1)
    else: raise ValueError(encoding)
    if backend=='statevector': return expectations_batch(states,n)
    return sample_expectations(states,SHOTS,NOISE_P,seed,depth)

def prep(seed,ncomp):
    d=load_breast_cancer(); X=d.data.astype(np.float64); y=d.target.astype(int)
    # sklearn target: 0 malignant, 1 benign. retain mapping as provided; balanced enough.
    idx=np.arange(len(y))
    tr, rem=train_test_split(idx,train_size=0.6,stratify=y,random_state=seed)
    va,te=train_test_split(rem,train_size=0.5,stratify=y[rem],random_state=seed+1)
    scaler=StandardScaler(); Xtr=scaler.fit_transform(X[tr]); Xv=scaler.transform(X[va]); Xte=scaler.transform(X[te])
    pca=PCA(n_components=8,random_state=seed); Xtrp=pca.fit_transform(Xtr); Xvp=pca.transform(Xv); Xtep=pca.transform(Xte)
    Xtrp=Xtrp[:,:ncomp]; Xvp=Xvp[:,:ncomp]; Xtep=Xtep[:,:ncomp]
    Xtra=angle_features(Xtrp,Xtrp); Xva=angle_features(Xvp,Xtrp); Xtea=angle_features(Xtep,Xtrp)
    return Xtra,y[tr],Xva,y[va],Xtea,y[te],{'train':len(tr),'val':len(va),'test':len(te),'explained_variance':float(pca.explained_variance_ratio_[:ncomp].sum()),'train_indices':tr.tolist(),'val_indices':va.tolist(),'test_indices':te.tolist()}

def fit_classical(Xtr,ytr,Xv,yv,Xte,yte,seed,model='lr'):
    if model=='lr': clf=LogisticRegression(max_iter=500,C=1.0,random_state=seed)
    else: clf=SVC(C=1.0,gamma='scale',kernel='rbf',random_state=seed)
    clf.fit(Xtr,ytr); return clf.predict(Xte)

def run_qfm(Xtr,ytr,Xv,yv,Xte,yte,n,seed,backend,encoding='angle_ry'):
    rng=np.random.default_rng(seed); params=rng.normal(0,.01,size=(DEPTH,n,2))
    trf=simulate_features(Xtr,params,encoding,backend,seed+1,n,DEPTH)
    vf=simulate_features(Xv,params,encoding,backend,seed+2,n,DEPTH)
    tef=simulate_features(Xte,params,encoding,backend,seed+3,n,DEPTH)
    W,b=fit_head(trf,ytr,seed)
    p,_=eval_head(vf,yv,W,b); pe,_=eval_head(tef,yte,W,b)
    return pe

def run_qk(Xtr,ytr,Xv,yv,Xte,yte,n,seed,encoding='angle_ry'):
    # kernel on quantum states; train SVC precomputed with validation selection over C
    rng=np.random.default_rng(seed); params=rng.normal(0,.01,size=(DEPTH,n,2))
    s_tr=statevector_angles(Xtr,params) if encoding=='angle_ry' else statevector_zz(Xtr)
    s_v=statevector_angles(Xv,params) if encoding=='angle_ry' else statevector_zz(Xv)
    s_te=statevector_angles(Xte,params) if encoding=='angle_ry' else statevector_zz(Xte)
    Ktr=np.abs(s_tr@s_tr.conj().T)**2; Kv=np.abs(s_v@s_tr.conj().T)**2; Kte=np.abs(s_te@s_tr.conj().T)**2
    best=None; bestacc=-1
    for C in [0.1,1,10,100]:
        clf=SVC(C=C,kernel='precomputed'); clf.fit(Ktr,ytr); pv=clf.predict(Kv); acc=accuracy_score(yv,pv)
        if acc>bestacc: bestacc=acc; best=C
    clf=SVC(C=best,kernel='precomputed'); clf.fit(Ktr,ytr); return clf.predict(Kte)

def save_row(rows, task, seed,n,model,backend,enc,pred,yte,extra=None):
    m=metrics(yte,pred); m.update({'task':task,'seed':seed,'n_components':n,'n_qubits':n,'depth':DEPTH,'model':model,'backend':backend,'encoding':enc,'shots':SHOTS if backend=='noisy_shot' else None,'noise_probability':NOISE_P if backend=='noisy_shot' else 0.0,'train_size':None,'validation_size':None,'test_size':len(yte)})
    if extra: m.update(extra)
    rows.append(m)

def main():
    rows=[]; pred_records=[]
    for seed in RNG_SEEDS:
        for n in NCOMP_LIST:
            Xtr,ytr,Xv,yv,Xte,yte,meta=prep(seed,n)
            # classical
            for mdl in ['lr','rbf']:
                pr=fit_classical(Xtr,ytr,Xv,yv,Xte,yte,seed,mdl); save_row(rows,'WDBC',seed,n,'Logistic Regression' if mdl=='lr' else 'RBF-SVM','statevector','classical',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
                if n==8: pred_records.append({'task':'WDBC','seed':seed,'n':n,'model':'Logistic Regression' if mdl=='lr' else 'RBF-SVM','backend':'statevector','y_true':yte.tolist(),'y_pred':pr.tolist()})
            # qfm, vqc, qk
            for backend in ['statevector','noisy_shot']:
                pr=run_qfm(Xtr,ytr,Xv,yv,Xte,yte,n,seed,backend); save_row(rows,'WDBC',seed,n,'Frozen QFM',backend,'angle_ry',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
                if n==8 and backend=='statevector': pred_records.append({'task':'WDBC','seed':seed,'n':n,'model':'Frozen QFM','backend':backend,'y_true':yte.tolist(),'y_pred':pr.tolist()})
                pr_v,_=run_vqc(Xtr,ytr,Xv,yv,Xte,yte,n,seed,backend,'angle_ry',DEPTH,80); save_row(rows,'WDBC',seed,n,'Trainable VQC',backend,'angle_ry',pr_v,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
                if n==8 and backend=='statevector': pred_records.append({'task':'WDBC','seed':seed,'n':n,'model':'Trainable VQC','backend':backend,'y_true':yte.tolist(),'y_pred':pr_v.tolist()})
            pr=run_qk(Xtr,ytr,Xv,yv,Xte,yte,n,seed,'angle_ry'); save_row(rows,'WDBC',seed,n,'Quantum Kernel SVM','statevector','angle_ry',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
            if n==8: pred_records.append({'task':'WDBC','seed':seed,'n':n,'model':'Quantum Kernel SVM','backend':'statevector','y_true':yte.tolist(),'y_pred':pr.tolist()})
    # Stage 3 encoding ablation: ZZ vs angle, n=8, QFM+QK, seeds 7/19/31, statevector only
    for seed in RNG_SEEDS:
        Xtr,ytr,Xv,yv,Xte,yte,meta=prep(seed,8)
        for enc in ['angle_ry','zz_feature']:
            pr=run_qfm(Xtr,ytr,Xv,yv,Xte,yte,8,seed,'statevector',enc); save_row(rows,'WDBC_EncodingAblation',seed,8,'Frozen QFM','statevector',enc,pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
            if enc=='zz_feature': pred_records.append({'task':'WDBC_EncodingAblation','seed':seed,'n':8,'model':'Frozen QFM','backend':'statevector','encoding':enc,'y_true':yte.tolist(),'y_pred':pr.tolist()})
            pr=run_qk(Xtr,ytr,Xv,yv,Xte,yte,8,seed,enc); save_row(rows,'WDBC_EncodingAblation',seed,8,'Quantum Kernel SVM','statevector',enc,pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
            if enc=='zz_feature': pred_records.append({'task':'WDBC_EncodingAblation','seed':seed,'n':8,'model':'Quantum Kernel SVM','backend':'statevector','encoding':enc,'y_true':yte.tolist(),'y_pred':pr.tolist()})
    # Stage 4 expansion: 7 new seeds, WDBC, 8D, statevector only for 5 model families; use same splits.
    for seed in EXTRA_SEEDS:
        Xtr,ytr,Xv,yv,Xte,yte,meta=prep(seed,8)
        for mdl in ['lr','rbf']:
            pr=fit_classical(Xtr,ytr,Xv,yv,Xte,yte,seed,mdl); save_row(rows,'WDBC_Replication10',seed,8,'Logistic Regression' if mdl=='lr' else 'RBF-SVM','statevector','classical',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
        pr=run_qfm(Xtr,ytr,Xv,yv,Xte,yte,8,seed,'statevector'); save_row(rows,'WDBC_Replication10',seed,8,'Frozen QFM','statevector','angle_ry',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
        pr,_=run_vqc(Xtr,ytr,Xv,yv,Xte,yte,8,seed,'statevector','angle_ry',DEPTH,80); save_row(rows,'WDBC_Replication10',seed,8,'Trainable VQC','statevector','angle_ry',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
        pr=run_qk(Xtr,ytr,Xv,yv,Xte,yte,8,seed,'angle_ry'); save_row(rows,'WDBC_Replication10',seed,8,'Quantum Kernel SVM','statevector','angle_ry',pr,yte,{'train_size':meta['train'],'validation_size':meta['val'],'explained_variance':meta['explained_variance']})
    pd.DataFrame(rows).to_csv(OUT/'experiment_results.csv',index=False)
    (OUT/'prediction_records.json').write_text(json.dumps(pred_records),encoding='utf-8')
    # summary on 3-seed external study
    df=pd.DataFrame(rows)
    # primary external only
    ext=df[(df.task=='WDBC') & (df.seed.isin(RNG_SEEDS))].copy()
    summ=ext.groupby(['model','n_components','backend'],dropna=False).agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index()
    summ.to_csv(OUT/'external_summary.csv',index=False)
    # encoding summary
    enc=df[df.task=='WDBC_EncodingAblation'].groupby(['model','encoding']).agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); enc.to_csv(OUT/'encoding_summary.csv',index=False)
    # 10-seed primary statevector summary: first 3 from WDBC plus 7 from WDBC_Replication10
    prim=pd.concat([df[(df.task=='WDBC') & (df.seed.isin(RNG_SEEDS)) & (df.n_components==8) & (df.backend=='statevector')],df[df.task=='WDBC_Replication10']])
    prim=prim.groupby('model').agg(runs=('accuracy','count'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std')).reset_index(); prim.to_csv(OUT/'replication10_summary.csv',index=False)
    # sensitivity for LR vs QK paired differences across 10 seeds
    accs=[]
    for seed in ALL10:
        sub=(df[(df.n_components==8)&(df.backend=='statevector')&(df.seed==seed)&(df.model.isin(['Logistic Regression','Quantum Kernel SVM']))])
        d1=sub.set_index('model')['accuracy'];
        if set(['Logistic Regression','Quantum Kernel SVM']).issubset(d1.index): accs.append(float(d1['Logistic Regression']-d1['Quantum Kernel SVM']))
    accs=np.array(accs); sd=float(accs.std(ddof=1)); mean=float(accs.mean())
    # paired n power via noncentral t, solve minimal n
    def power(n, delta=0.05, sd=sd):
        dfree=n-1; nc=delta*math.sqrt(n)/sd; crit=student_t.ppf(1-0.05/2,dfree)
        return 1-(nct.cdf(crit,dfree,nc)-nct.cdf(-crit,dfree,nc))
    sens=[]
    for delta in [0.03,0.05,0.10]:
        for target in [.8,.9]:
            n=2
            while n<5000 and power(n,delta)<target: n+=1
            sens.append({'delta_accuracy':delta,'target_power':target,'paired_sd':sd,'n_required':n,'achieved_sample_seeds':len(accs),'observed_mean_signed_difference_LR_minus_QK':mean})
    pd.DataFrame(sens).to_csv(OUT/'sensitivity_analysis.csv',index=False)
    # confusion matrices: 10-seed aggregate for statevector 8D, each model
    cms={}
    for model in ['Logistic Regression','RBF-SVM','Frozen QFM','Trainable VQC','Quantum Kernel SVM']:
        all_true=[]; all_pred=[]
        for rec in pred_records:
            if rec['task']=='WDBC' and rec['n']==8 and rec['backend']=='statevector' and rec['model']==model:
                all_true+=rec['y_true']; all_pred+=rec['y_pred']
        if all_true: cms[model]=confusion_matrix(all_true,all_pred,labels=[0,1]).tolist()
    (OUT/'confusion_matrices.json').write_text(json.dumps(cms,indent=2),encoding='utf-8')
    # metadata
    meta={'external_dataset':'Breast Cancer Wisconsin (Diagnostic)','instances':569,'features':30,'split_rule':'60/20/20 stratified; validation/test split uses seed+1','seeds_primary':RNG_SEEDS,'seeds_replication_added':EXTRA_SEEDS,'pca_components':[4,8],'depth':DEPTH,'noise_probability':NOISE_P,'shots':SHOTS,'vqc_iterations':80,'unique_rows_written':int(len(df))}
    (OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Rows',len(df)); print(summ.to_string(index=False)); print('\nEncoding'); print(enc.to_string(index=False)); print('\nReplication'); print(prim.to_string(index=False)); print('\nSensitivity',pd.DataFrame(sens).to_string(index=False))

if __name__=='__main__': main()
