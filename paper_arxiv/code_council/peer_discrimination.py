"""Independent rebuild of the 37-case verdict table. Reads only raw judgment files."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, rankdata
# usage: python peer_discrimination.py [REPO] [OUT]; defaults: this repository, ./out
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
R = REPO/"results"
rng = np.random.default_rng(12345)
FRONT = ["GPT","Claude","Gemini","Grok","DeepSeek"]; OW = ["Qwen","Llama","Mistral","Phi"]

def jl(p):
    recs=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
    n_all=len(recs); recs=[r for r in recs if r.get("ok")]
    d={r["item_id"]:r for r in recs}
    return list(d.values()), n_all

def attr_from_jsonl(p, single):
    recs,n_all=jl(p); rows=[]
    for r in recs:
        if single:
            pr=r.get("parsed"); pred=pr.get("model") if isinstance(pr,dict) else None
            rows.append((r["prompt_id"],r["judge"],r["true_author"],pred))
        else:
            pr=r.get("parsed") if isinstance(r.get("parsed"),dict) else {}
            for s,a in r["slot_to_author"].items():
                rows.append((r["prompt_id"],r["judge"],a,(pr.get(s) or {}).get("model")))
    return pd.DataFrame(rows,columns=["prompt","judge","author","pred"]), n_all, len(recs)

def attr_from_csv(p):
    d=pd.read_csv(p); n=len(d)
    d=d[d.prompt_id!="M5"]
    d=d.rename(columns={"prompt_id":"prompt","true_author":"author","guessed_model":"pred"})
    return d[["prompt","judge","author","pred"]].copy(), n, len(d)

def yn_from_jsonl(p):
    recs,n_all=jl(p)
    d=pd.DataFrame(recs).rename(columns={"prompt_id":"prompt","true_author":"author"})
    return d[["prompt","judge","author","says_yes","p_yes"]].copy(), n_all, len(recs)

def pb_pmf(q):
    pmf=np.array([1.0])
    for x in q: pmf=np.convolve(pmf,[1-x,x])
    return pmf

def exact_two_sided(q, s_obs, stat):
    """S ~ PoissonBinomial(q); stat monotone in S. two-sided |stat|>=|obs| and one-sided upper."""
    pmf=pb_pmf(q); sup=stat(np.arange(len(pmf))); obs=stat(s_obs)
    return float(min(1,pmf[np.abs(sup)>=abs(obs)-1e-12].sum())), float(min(1,pmf[sup>=obs-1e-12].sum()))

def holm(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); m=len(ps); out=np.empty(m); run=0
    for i,k in enumerate(o):
        run=max(run,min(1,(m-i)*ps[k])); out[k]=run
    return out

def cube(a, names, judge_target):
    """N[p, author, judge] = 1 if judge named judge_target on (p, author's text)."""
    prompts=sorted(a.prompt.unique()); pi={p:i for i,p in enumerate(prompts)}; mi={m:i for i,m in enumerate(names)}
    N=np.full((len(prompts),len(names),len(names)),np.nan)
    for r in a.itertuples():
        N[pi[r.prompt],mi[r.author],mi[r.judge]]=float(r.pred==judge_target)
    return N

def attribution_case(a, names, J):
    k=len(names); j=names.index(J)
    N=cube(a,names,J)
    assert not np.isnan(N).any(), "unbalanced"
    G=N.shape[0]
    own_self=N[:,j,j]               # J names J on own text
    hits=int(own_self.sum()); n=G
    fa_mat=np.delete(N[:,:,j],j,axis=1)   # J names J on others' texts  (G, k-1)
    fa=fa_mat.mean(); 
    peer_mat=np.delete(N[:,j,:],j,axis=1) # peers name J on J's text   (G, k-1)
    peer=peer_mat.mean()
    # standard test
    p_std=binomtest(hits,n,1/k,alternative="greater").pvalue
    # discrimination: within-prompt relabel which of J's k judged texts is its own -> exact
    V=N[:,:,j]; tot=V.sum()
    st=lambda s: s/G-(tot-s)/(G*(k-1))
    p_disc,p_disc_1=exact_two_sided(V.mean(axis=1),hits,st)
    # self-advantage: within-response relabel which judge is self -> exact
    M=N[:,j,:]; tot2=M.sum()
    st2=lambda s: s/G-(tot2-s)/(G*(k-1))
    p_adv,p_adv_1=exact_two_sided(M.mean(axis=1),hits,st2)
    # (g) peers' discrimination for naming J: peer hit on J's texts minus peer FA naming J on non-J texts
    # variant A: all non-J texts (incl. the peer's own); variant B: excluding the peer's own texts
    peers=[i for i in range(k) if i!=j]; others=peers
    dJ_p = N[:,j,j]-N[:,others,j].mean(axis=1)                       # per prompt, judge J
    dP_p = np.mean([N[:,j,q]-N[:,others,q].mean(axis=1) for q in peers],axis=0)
    dPB_p= np.mean([N[:,j,q]-N[:,[o for o in others if o!=q],q].mean(axis=1) for q in peers],axis=0)
    peer_fa=np.mean([N[:,others,q].mean() for q in peers])
    peer_faB=np.mean([N[:,[o for o in others if o!=q],q].mean() for q in peers])
    def signflip(delta):
        obs=delta.mean()
        S=rng.choice([-1.0,1.0],size=(100000,len(delta)))
        sims=(S*delta).mean(axis=1)
        return float(((np.abs(sims)>=abs(obs)-1e-12).sum()+1)/(100001))
    def bci(delta):
        idx=rng.integers(0,len(delta),size=(10000,len(delta)))
        b=delta[idx].mean(axis=1); return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
    dd=dJ_p-dP_p; ddB=dJ_p-dPB_p
    lo,hi=bci(dd)
    return dict(n_own=n,self_hits=hits,self_rate=hits/n,chance=1/k,p_std=p_std,
        fa_rate=fa,disc=hits/n-fa,p_disc=p_disc,
        peer_rate=peer,adv=hits/n-peer,p_adv=p_adv,
        peer_fa=peer_fa,peer_disc=peer-peer_fa,dd=float(dd.mean()),dd_lo=lo,dd_hi=hi,p_dd=signflip(dd),
        peer_discB=peer-peer_faB,ddB=float(ddB.mean()),p_ddB=signflip(ddB))

def yesno_case(d, names, J):
    k=len(names); m=d[d.judge==J]
    piv=m.pivot(index="prompt",columns="author",values="says_yes").astype(float)[names]
    pv=m.pivot(index="prompt",columns="author",values="p_yes").astype(float)[names]
    assert not piv.isna().any().any()
    j=names.index(J); V=piv.to_numpy(); G=len(V)
    hits=int(V[:,j].sum()); tot=V.sum()
    fa=np.delete(V,j,axis=1).mean()
    p_std=binomtest(hits,G,0.5,alternative="greater").pvalue
    st=lambda s: s/G-(tot-s)/(G*(k-1))
    p_disc,_=exact_two_sided(V.mean(axis=1),hits,st)
    # AUC on p_yes with within-prompt permutation (authors' criterion)
    P=pv.to_numpy(); rk=rankdata(P.ravel()).reshape(P.shape); n1,n0=G,G*(k-1)
    auc=(rk[:,j].sum()-n1*(n1+1)/2)/(n1*n0)
    Rr=rng.integers(0,k,size=(200000,G))
    pa=(rk[np.arange(G)[None,:],Rr].sum(axis=1)-n1*(n1+1)/2)/(n1*n0)
    p_auc=float(((np.abs(pa-.5)>=abs(auc-.5)-1e-12).sum()+1)/200001)
    return dict(n_own=G,self_hits=hits,self_rate=hits/G,chance=0.5,p_std=p_std,
        fa_rate=fa,disc=hits/G-fa,p_disc=p_disc,auc=auc,p_auc=p_auc)

SET=[]
def add(setting,fmt,names,kind,df,n_all,n_ok):
    print(f"{setting:38s} rows_in_file={n_all} ok={n_ok} judgments={len(df)} unparsed_pred={int(df['pred'].isna().sum()) if 'pred' in df else 'n/a'}")
    for J in names:
        r=attribution_case(df,names,J) if kind=="attr" else yesno_case(df,names,J)
        r.update(setting=setting,format=fmt,judge=J,kind=kind); SET.append(r)

add("frontier chat-app, lineup","lineup",FRONT,"attr",*attr_from_csv(R/"judgments.csv"))
add("frontier chat-app, single text","single",FRONT,"attr",*attr_from_csv(R/"judgments_single.csv"))
add("frontier chat-app, yes/no","yesno",FRONT,"yn",*yn_from_jsonl(R/"revision/frontier/binary.jsonl"))
add("frontier API, lineup","lineup",FRONT,"attr",*attr_from_jsonl(R/"revision/frontier/lineup_api120.jsonl",False))
add("frontier API, single text","single",FRONT,"attr",*attr_from_jsonl(R/"revision/frontier/single_api120.jsonl",True))
add("open-weight, lineup","lineup",OW,"attr",*attr_from_jsonl(R/"revision/openweight/lineup.jsonl",False))
add("open-weight, single text","single",OW,"attr",*attr_from_jsonl(R/"revision/openweight/single.jsonl",True))
add("open-weight, yes/no","yesno",OW,"yn",*yn_from_jsonl(R/"revision/openweight/binary.jsonl"))
T=pd.DataFrame(SET)
cols=["setting","judge","kind","n_own","self_hits","self_rate","chance","p_std","fa_rate","disc","p_disc","auc","p_auc","peer_rate","adv","p_adv","peer_fa","peer_disc","dd","dd_lo","dd_hi","p_dd","peer_discB","ddB","p_ddB"]
T=T[cols]
# yes/no: the paper's discrimination criterion is the AUC permutation test
T["p_disc_paper"]=np.where(T.kind=="yn",T.p_auc,T.p_disc)
T["disc_pos_paper"]=np.where(T.kind=="yn",T.auc>0.5,T.disc>0)
for c in ["p_std","p_disc","p_disc_paper","p_adv","p_dd","p_ddB"]:
    T[c+"_holmW"]=np.nan; T[c+"_holmAll"]=np.nan
    for s,g in T.groupby("setting"):
        ok=g[c].notna()
        if ok.any(): T.loc[g.index[ok],c+"_holmW"]=holm(g.loc[ok,c])
    ok=T[c].notna(); T.loc[ok,c+"_holmAll"]=holm(T.loc[ok,c])
T.to_pickle(OUT/"cases.pkl"); T.to_csv(OUT/"cases_raw.csv",index=False)
print(T[["setting","judge","self_rate","p_std","p_std_holmW","disc","p_disc","adv","p_adv"]].to_string())
