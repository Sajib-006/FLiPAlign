import json,glob,numpy as np,collections,sys
def m(p):
    o={}
    for f in glob.glob(p):
        for r in json.load(open(f)): o[r['case']]=r
    return o
def ad(a,b): return abs((a-b+180)%360-180)
def in_zone(r):
    if not r['certified'] or 'true_shift_units' not in r: return None
    dz,rz=r['finest_zone']; tdeg,tfl=r['truth']
    return tfl==r['flip'] and ad(r['angle'],tdeg)<=dz and (rz is None or np.hypot(*(np.array(r['shift_units'])-r['true_shift_units']))<=rz)
def margin(st,k=500):
    st=st[:k]; b=max(st,key=lambda s:s['score'])
    far=[s['score'] for s in st if s['flip']!=b['flip'] or ad(s['angle'],b['angle'])>20 or np.hypot(*(np.array(s['shift'])-b['shift']))>30]
    return b['score']-max(far), b['tre_um']
def report(setname, man, groups):
    print(f"\n===== {setname} =====")
    runs={k:m(f'{setname}{k}_t*.json') for k in ('ov','h2')}
    L=m(f'{setname}local_t*.json') or m(f'{setname}local500_t*.json')
    base=collections.defaultdict(dict)
    for f in glob.glob(f'{setname}base_t*.json')+glob.glob(f'{setname}slat_t*.json')+glob.glob(f'{setname}base3_t*.json'):
        for r in json.load(open(f)):
            if 'tre_um' in r: base[(r['method'],r['fit'])][r['case']]=r['tre_um']
    for g in groups+['ALL']:
        cs=sorted(c for c in man if g=='ALL' or man[c]==g)
        print(f"-- {g} (n={len(cs)})")
        for k,O in runs.items():
            cc=[c for c in cs if c in O]
            if not cc: continue
            right=sum(O[c]['tre_um']<200 for c in cc); cert=[c for c in cc if O[c]['certified']]
            wc=sum(O[c]['tre_um']>=200 for c in cert); outz=sum(in_zone(O[c]) is False for c in cert)
            print(f"   {'per-overlap' if k=='ov' else 'original   '}: done {len(cc)}, right {right}, certified {len(cert)}, certified-wrong {wc}, truth-outside-zone {outz}, median err {np.median([O[c]['tre_um'] for c in cc]):.0f}")
        mt=[margin(L[c]['starts']) for c in cs if c in L]
        if mt:
            ans=[x for x in mt if x[0]>0.01]; ans2=[x for x in mt if x[0]>0.02]
            print(f"   local500   : right {sum(x[1]<200 for x in mt)}; refuse-rule tau .01 answered {len(ans)} wrong {sum(x[1]>=200 for x in ans)}; tau .02 answered {len(ans2)} wrong {sum(x[1]>=200 for x in ans2)}")
        best={}
        for (meth,fit),d in base.items():
            v=[d[c] for c in cs if c in d]
            if len(v)<len(cs)*0.8: continue
            nm=meth.split('_')[0]; med=np.median(v)
            if nm not in best or med<best[nm][0]: best[nm]=(med,np.mean(np.array(v)<200),meth+'/'+fit)
        print('   baselines  : '+'; '.join(f"{nm} {med:.0f}um right {fr:.0%}" for nm,(med,fr,_) in sorted(best.items(),key=lambda x:x[1][0])))
man1={c:'breast' for c in range(24)}
man2={x['case']:x['tissue'] for x in json.load(open('gt2/manifest.json'))}
man3={x['case']:x['tissue'] for x in json.load(open('gt3/manifest.json'))}
report('gt1',man1,['breast'])
report('gt2',man2,['colon','mouse brain','DLPFC'])
report('gt3',man3,sorted(set(man3.values())))
