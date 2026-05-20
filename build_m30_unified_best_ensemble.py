#!/usr/bin/env python3
"""M30 unified best ensemble."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
EPS=1e-12; B0=0.5215; B1=0.8101
TC=["true_mapped_label","true_idx","y_true","mapped_label"]

def ensure(p):
    p.mkdir(parents=True, exist_ok=True)

def sj(o,p):
    with open(p,"w",encoding="utf-8") as f:
        json.dump(o,f,indent=2,ensure_ascii=False)

def npz(p):
    p=np.asarray(p,dtype=float); p=np.clip(p,EPS,1); return p/p.sum(1,keepdims=True)

def lg(p):
    return np.log(npz(p)+EPS)

def sm(x):
    x=x-x.max(1,keepdims=True); e=np.exp(x); return e/e.sum(1,keepdims=True)

def met(y,pr,nc):
    a=accuracy_score(y,pr)
    _,_,fp,_=precision_recall_fscore_support(y,pr,average="macro",zero_division=0)
    la=list(range(nc))
    _,_,fa,_=precision_recall_fscore_support(y,pr,labels=la,average="macro",zero_division=0)
    _,_,fw,_=precision_recall_fscore_support(y,pr,average="weighted",zero_division=0)
    return dict(accuracy=float(a),macro_f1_present=float(fp),macro_f1_all=float(fa),weighted_f1=float(fw))

def tc(df):
    for c in TC:
        if c in df.columns:
            return c
    raise ValueError(list(df.columns))

class E:
    def __init__(self,n,p,y,c):
        self.name,self.probs,self.y,self.csv=n,p,y,c

def load(n,pp,cp):
    p=npz(np.load(pp)); d=pd.read_csv(cp); y=d[tc(d)].astype(int).to_numpy()
    if len(y)!=len(p):
        raise RuntimeError(n)
    return E(n,p,y,cp)
def pmap(root):
    m22=root/"M22_selective_ensemble"
    m26=root/"M26_calibrated_context_ensemble"/"run_v1"
    m28=root/"M28_error_driven_soft_ensemble"/"run_v1"
    v=m22/"val_predictions"; t=m22/"test_predictions"; e=m22/"ensemble_run_v1"
    return dict(
        m19=(v/"m19_val_probs.npy",t/"m19_val_probs.npy",v/"m19_val_predictions.csv",t/"m19_val_predictions.csv"),
        m21=(v/"m21_val_probs.npy",t/"m21_val_probs.npy",v/"m21_val_predictions.csv",t/"m21_val_predictions.csv"),
        m22=(e/"val_ensemble_probs.npy",e/"test_ensemble_probs.npy",e/"val_ensemble_predictions.csv",e/"test_ensemble_predictions.csv"),
        m26=(m26/"val_m26_probs.npy",m26/"test_m26_probs.npy",m26/"val_m26_predictions.csv",m26/"test_m26_predictions.csv"),
        m28=(m28/"val_m28_probs.npy",m28/"test_m28_probs.npy",m28/"val_m28_predictions.csv",m28/"test_m28_predictions.csv"),
    )

def prior(csv,nc,sm):
    d=pd.read_csv(csv)
    if "mapped_label" in d.columns:
        y=d["mapped_label"].astype(int).to_numpy()
    else:
        y=d["pill_label"].map({u:i for i,u in enumerate(sorted(d["pill_label"].unique()))}).astype(int).to_numpy()
    c=np.zeros(nc)
    for i in y:
        if 0<=int(i)<nc:
            c[int(i)]+=1
    c+=sm
    return c/c.sum()

def wgrid(names,step):
    vals=np.round(np.arange(0,1+1e-9,step),4); out=[]; n=len(names)
    if n==2:
        for a in vals:
            b=round(1-float(a),4)
            if b>=0:
                out.append({names[0]:float(a),names[1]:float(b)})
        return out
    def rec(pre,d):
        if d==n-1:
            last=round(1-sum(pre),4)
            if last>=0:
                out.append(dict(zip(names,pre+[last])))
            return
        for v in vals:
            if sum(pre)+v<=1+1e-9:
                rec(pre+[float(v)],d+1)
    rec([],0)
    return out

def fuse(exs,w,pr,a,t):
    L=None
    for e in exs:
        ww=float(w.get(e.name,0))
        if ww<=0:
            continue
        l=lg(e.probs)*ww
        L=l if L is None else L+l
    lp=np.log(np.clip(pr,EPS,1))[None,:]
    return sm((L-a*lp)/max(t,1e-6))

def sf(val,nc,pr,step,al,te,subs):
    em={e.name:e for e in val}; y=val[0].y; rows=[]; ns=[e.name for e in val]
    for sub in subs:
        sub=[n for n in sub if n in em]
        if len(sub)<2:
            continue
        exs=[em[n] for n in sub]
        for wg in wgrid(sub,step):
            for alpha in al:
                for temp in te:
                    p=fuse(exs,wg,pr,alpha,temp)
                    pred=p.argmax(1)
                    m=met(y,pred,nc)
                    rows.append(dict(expert_subset=",".join(sub),temperature=temp,logit_adjust_alpha=alpha,**{f"w_{k}":wg.get(k,0) for k in ns},**m))
    g=pd.DataFrame(rows).sort_values(["macro_f1_present","macro_f1_all","accuracy"],ascending=False).reset_index(drop=True)
    return g,g.iloc[0].to_dict()
def cstats(y,pred,nc):
    pr,re,f1,su=precision_recall_fscore_support(y,pred,labels=list(range(nc)),average=None,zero_division=0)
    ps=np.bincount(pred,minlength=nc)
    return pd.DataFrame(dict(support=su.astype(int),pred_support=ps.astype(int),precision=pr,recall=re,f1=f1))

def bvec(st,nc,a,b,g,d,ss,clip):
    s=st.support.to_numpy(float); ps=st.pred_support.to_numpy(float)
    pr=st.precision.to_numpy(float); re=st.recall.to_numpy(float)
    present=s>0
    pv=(s+ss)/(s.sum()+ss*nc)
    pb=-np.log(np.clip(pv/(1/nc),EPS,None))
    rg=np.where(present,1-re,0); pg=np.where(ps>0,1-pr,0)
    fp=np.log1p(np.maximum(ps-s,0)); fp=fp/np.clip(fp.max(),1,None)
    bias=a*pb+b*rg-g*pg-d*fp
    bias=np.where(present,bias,np.minimum(bias,0))
    bias=np.clip(bias,-clip,clip)
    return bias-bias.mean()

def calib(p,temp,bias):
    x=np.log(np.clip(npz(p),EPS,None))/float(temp)
    return sm(x+bias.reshape(1,-1))

def sc(y,p,nc,drop):
    bp=p.argmax(1); bm=met(y,bp,nc); st=cstats(y,bp,nc)
    bs,br,bb=-1e18,None,None; rows=[]
    for temp in [0.75,0.85,1.0,1.15]:
        for a in [0,0.15,0.3,0.45,0.6]:
            for b in [0,0.25,0.5,0.75]:
                for g in [0,0.2,0.35]:
                    for d in [0,0.15,0.3]:
                        bias=bvec(st,nc,a,b,g,d,3,1)
                        cp=calib(p,temp,bias)
                        pred=cp.argmax(1)
                        m=met(y,pred,nc)
                        sc2=m["macro_f1_present"]-max(0,bm["accuracy"]-m["accuracy"]-drop)*5
                        row=dict(temperature=temp,alpha_prior=a,beta_recall=b,gamma_precision_penalty=g,delta_false_positive_penalty=d,selection_score=sc2,**m)
                        rows.append(row)
                        if sc2>bs:
                            bs,br,bb=sc2,row,bias.copy()
    return pd.DataFrame(rows).sort_values("selection_score",ascending=False),br,bb

def save(od,sp,df,y,p,cfg):
    pred=p.argmax(1); m=met(y,pred,p.shape[1])
    out=df.copy(); t=tc(out)
    out["m30_pred_mapped_label"]=pred
    out["m30_confidence"]=p.max(1)
    out["m30_is_correct"]=(out[t].astype(int).values==pred).astype(int)
    out.to_csv(od/f"{sp}_m30_predictions.csv",index=False)
    np.save(od/f"{sp}_m30_probs.npy",p.astype(np.float32))
    sj(dict(metrics=m,config=cfg),od/f"{sp}_m30_metrics.json")
    return m

def cand(repo,pred,candp,out):
    sc=repo/"evaluate_candidate_benchmarks.py"
    if sc.exists():
        subprocess.run([sys.executable,str(sc),"--candidates_csv",str(candp),"--m17_predictions",str(pred),"--output_dir",str(out)],check=True)

def prep(m,s):
    p=m["macro_f1_present"]
    r=dict(test_macro_f1_present=p,beats_paper_baseline_0_5215=p>B0,beats_paper_pika_full_0_8101_on_test_present=p>B1,
           gap_to_baseline=round(p-B0,6),gap_to_pika_full=round(p-B1,6),paper_reference=dict(baseline_f1=B0,pika_full_f1=B1))
    if s is not None:
        r["subset38_macro_f1_all_candidate"]=s
        r["beats_pika_full_on_subset38"]=s>B1
        r["gap_subset38_to_pika_full"]=round(s-B1,6)
    return r
def cstats(y,pred,nc):
    pr,re,f1,su=precision_recall_fscore_support(y,pred,labels=list(range(nc)),average=None,zero_division=0)
    ps=np.bincount(pred,minlength=nc)
    return pd.DataFrame(dict(support=su.astype(int),pred_support=ps.astype(int),precision=pr,recall=re,f1=f1))

def bvec(st,nc,a,b,g,d,ss,clip):
    s=st.support.to_numpy(float); ps=st.pred_support.to_numpy(float)
    pr=st.precision.to_numpy(float); re=st.recall.to_numpy(float)
    present=s>0
    pv=(s+ss)/(s.sum()+ss*nc)
    pb=-np.log(np.clip(pv/(1/nc),EPS,None))
    rg=np.where(present,1-re,0); pg=np.where(ps>0,1-pr,0)
    fp=np.log1p(np.maximum(ps-s,0)); fp=fp/np.clip(fp.max(),1,None)
    bias=a*pb+b*rg-g*pg-d*fp
    bias=np.where(present,bias,np.minimum(bias,0))
    bias=np.clip(bias,-clip,clip)
    return bias-bias.mean()

def calib(p,temp,bias):
    x=np.log(np.clip(npz(p),EPS,None))/float(temp)
    return sm(x+bias.reshape(1,-1))

def sc(y,p,nc,drop):
    bp=p.argmax(1); bm=met(y,bp,nc); st=cstats(y,bp,nc)
    bs,br,bb=-1e18,None,None; rows=[]
    for temp in [0.75,0.85,1.0,1.15]:
        for a in [0,0.15,0.3,0.45,0.6]:
            for b in [0,0.25,0.5,0.75]:
                for g in [0,0.2,0.35]:
                    for d in [0,0.15,0.3]:
                        bias=bvec(st,nc,a,b,g,d,3,1)
                        cp=calib(p,temp,bias)
                        pred=cp.argmax(1)
                        m=met(y,pred,nc)
                        sc2=m["macro_f1_present"]-max(0,bm["accuracy"]-m["accuracy"]-drop)*5
                        row=dict(temperature=temp,alpha_prior=a,beta_recall=b,gamma_precision_penalty=g,delta_false_positive_penalty=d,selection_score=sc2,**m)
                        rows.append(row)
                        if sc2>bs:
                            bs,br,bb=sc2,row,bias.copy()
    return pd.DataFrame(rows).sort_values("selection_score",ascending=False),br,bb

def save(od,sp,df,y,p,cfg):
    pred=p.argmax(1); m=met(y,pred,p.shape[1])
    out=df.copy(); t=tc(out)
    out["m30_pred_mapped_label"]=pred
    out["m30_confidence"]=p.max(1)
    out["m30_is_correct"]=(out[t].astype(int).values==pred).astype(int)
    out.to_csv(od/f"{sp}_m30_predictions.csv",index=False)
    np.save(od/f"{sp}_m30_probs.npy",p.astype(np.float32))
    sj(dict(metrics=m,config=cfg),od/f"{sp}_m30_metrics.json")
    return m

def cand(repo,pred,candp,out):
    sc=repo/"evaluate_candidate_benchmarks.py"
    if sc.exists():
        subprocess.run([sys.executable,str(sc),"--candidates_csv",str(candp),"--m17_predictions",str(pred),"--output_dir",str(out)],check=True)

def prep(m,s):
    p=m["macro_f1_present"]
    r=dict(test_macro_f1_present=p,beats_paper_baseline_0_5215=p>B0,beats_paper_pika_full_0_8101_on_test_present=p>B1,
           gap_to_baseline=round(p-B0,6),gap_to_pika_full=round(p-B1,6),paper_reference=dict(baseline_f1=B0,pika_full_f1=B1))
    if s is not None:
        r["subset38_macro_f1_all_candidate"]=s
        r["beats_pika_full_on_subset38"]=s>B1
        r["gap_subset38_to_pika_full"]=round(s-B1,6)
    return r
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_root",default="/content/drive/MyDrive/model")
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--train_csv",default="/content/drive/MyDrive/vaipe_splits/clean_paper_like_split_v2/train_clean.csv")
    ap.add_argument("--candidates_csv",default="/content/drive/MyDrive/model/audit_pika_protocol_v1/audit_candidate_benchmarks.csv")
    ap.add_argument("--num_classes",type=int,default=108)
    ap.add_argument("--weight_step",type=float,default=0.05)
    ap.add_argument("--prior_smoothing",type=float,default=1.0)
    ap.add_argument("--max_accuracy_drop_calib",type=float,default=0.005)
    ap.add_argument("--skip_calibration",action="store_true")
    ap.add_argument("--experts",default="m26,m28,m19,m21")
    ap.add_argument("--expert_subsets",default="m26,m28;m19,m21,m26;m19,m21,m26,m28;m19,m21,m22,m26,m28")
    ap.add_argument("--logit_adjust_alphas",default="-0.25,0.0,0.25,0.5")
    ap.add_argument("--temperatures",default="0.75,1.0,1.25")
    ap.add_argument("--repo_root",default=".")
    a=ap.parse_args()
    od=Path(a.output_dir); ensure(od); pm=pmap(Path(a.model_root))
    names=[x.strip() for x in a.experts.split(",") if x.strip()]
    subs=[[y.strip() for y in g.split(",") if y.strip()] for g in a.expert_subsets.split(";")]
    al=[float(x) for x in a.logit_adjust_alphas.split(",")]
    te=[float(x) for x in a.temperatures.split(",")]
    print("=== M30 UNIFIED BEST ENSEMBLE ===")
    val=[]; tst=[]
    for n in names:
        vp,tp,vc,tc_path=pm[n]
        val.append(load(n,vp,vc)); tst.append(load(n,tp,tc_path))
        print(n,val[-1].probs.shape)
    nc=val[0].probs.shape[1]
    pr=prior(Path(a.train_csv),nc,a.prior_smoothing)
    base=[]
    for sn,bu in [("val",val),("test",tst)]:
        y=bu[0].y
        for e in bu:
            base.append(dict(split=sn,model=e.name.upper(),**met(y,e.probs.argmax(1),nc)))
    bdf=pd.DataFrame(base); bdf.to_csv(od/"m30_baseline_metrics.csv",index=False); print(bdf)
    grid,best=sf(val,nc,pr,a.weight_step,al,te,subs)
    grid.to_csv(od/"m30_fusion_val_grid.csv",index=False); print(grid.head(10))
    sub=[x.strip() for x in str(best["expert_subset"]).split(",") if x.strip()]
    wg={n:float(best.get(f"w_{n}",0)) for n in sub}
    fcfg=dict(expert_subset=sub,weights=wg,logit_adjust_alpha=float(best["logit_adjust_alpha"]),temperature=float(best["temperature"]))
    sj(fcfg,od/"m30_fusion_best_config.json")
    vs=[e for e in val if e.name in sub]; ts=[e for e in tst if e.name in sub]
    vp=fuse(vs,wg,pr,fcfg["logit_adjust_alpha"],fcfg["temperature"])
    tp=fuse(ts,wg,pr,fcfg["logit_adjust_alpha"],fcfg["temperature"])
    save(od,"val",pd.read_csv(vs[0].csv),vs[0].y,vp,fcfg)
    tmeta=pd.read_csv(ts[0].csv); tm=save(od,"test",tmeta,ts[0].y,tp,fcfg)
    if not a.skip_calibration:
        cg,cb,bb=sc(vs[0].y,vp,nc,a.max_accuracy_drop_calib)
        cg.to_csv(od/"m30_calibration_val_grid.csv",index=False)
        np.save(od/"m30_calibration_bias.npy",bb.astype(np.float32))
        sj(cb,od/"m30_calibration_best_config.json")
        tm=save(od,"test_calibrated",tmeta,ts[0].y,calib(tp,float(cb["temperature"]),bb),{**fcfg,"calibration":cb})
    pred=od/"test_calibrated_m30_predictions.csv"
    if not pred.exists():
        pred=od/"test_m30_predictions.csv"
    cd=od/"candidate_eval"; ensure(cd)
    cand(Path(a.repo_root).resolve(),pred,Path(a.candidates_csv),cd)
    s38=None
    cc=cd/"candidate_benchmark_results.csv"
    if cc.exists():
        cdf=pd.read_csv(cc)
        mask=cdf.candidate.astype(str).str.contains("train_support_ge_100",na=False)
        if mask.any():
            s38=float(cdf.loc[mask].iloc[0].macro_f1_all_candidate)
    rep=prep(tm,s38)
    sj(rep,od/"m30_paper_target_report.json")
    pd.DataFrame([{**tm,"split":"test"}]).to_csv(od/"m30_summary.csv",index=False)
    print(json.dumps(rep,indent=2,ensure_ascii=False))
    print("Done",od)

if __name__=="__main__":
    main()