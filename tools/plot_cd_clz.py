#!/usr/bin/env python3
"""Plot Anabasis CD_vector and CL_z_vector, optionally overlaid with FEATFLOW BenchValues.txt.

Benchmark mapping used here:
  BenchValues Drag -> Anabasis CD_vector
  BenchValues Lift -> Anabasis CL_z_vector
because this mesh has the cylinder axis rotated relative to the FEATFLOW convention.
"""
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt


def fnum(x):
    return float(str(x).replace('D','E').replace('d','E'))


def read_sim(path, cl_col='CL_z_vector'):
    path=Path(path)
    with path.open(newline='') as f:
        r=csv.DictReader(f)
        cols=r.fieldnames or []
        for need in ['time','CD_vector',cl_col]:
            if need not in cols:
                raise SystemExit(f'Missing column {need} in {path}. Available: {cols}')
        t=[]; cd=[]; cl=[]
        for row in r:
            try:
                t.append(fnum(row['time'])); cd.append(fnum(row['CD_vector'])); cl.append(fnum(row[cl_col]))
            except Exception:
                pass
    if not t: raise SystemExit(f'No rows read from {path}')
    return t,cd,cl


def read_bench(path, lift_col='Lift'):
    path=Path(path)
    lines=path.read_text(errors='ignore').replace('D','E').replace('d','E').splitlines()
    header=None; rows=[]
    for line in lines:
        s=line.strip()
        if not s or s.startswith('#'): continue
        parts=s.split()
        if any(p.lower()=='time' for p in parts):
            header=parts; continue
        try:
            vals=[float(p) for p in parts]
        except Exception:
            continue
        if len(vals)>=3: rows.append(vals)
    if not rows: raise SystemExit(f'No numeric benchmark rows read from {path}')
    if header is None: header=['Time','Drag','Lift','ZForce']
    def idx(name, default):
        for i,h in enumerate(header):
            if h.lower()==name.lower(): return i
        return default
    it=idx('Time',0); idrag=idx('Drag',1); ilift=idx(lift_col,2)
    t=[]; cd=[]; cl=[]
    for r in rows:
        if len(r)>max(it,idrag,ilift):
            t.append(r[it]); cd.append(r[idrag]); cl.append(r[ilift])
    return t,cd,cl


def crop(t,cd,cl,tmin,tmax):
    out=[(a,b,c) for a,b,c in zip(t,cd,cl) if (tmin is None or a>=tmin) and (tmax is None or a<=tmax)]
    if not out: raise SystemExit('No data after tmin/tmax cropping')
    return [x[0] for x in out],[x[1] for x in out],[x[2] for x in out]


def stats(name,t,y):
    imin=min(range(len(y)), key=lambda i:y[i]); imax=max(range(len(y)), key=lambda i:y[i])
    print(f'{name}:')
    print(f'  min = {y[imin]: .12e} at t = {t[imin]:.8e}')
    print(f'  max = {y[imax]: .12e} at t = {t[imax]:.8e}')
    print(f'  end = {y[-1]: .12e} at t = {t[-1]:.8e}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('sim_csv')
    ap.add_argument('--bench', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--clean-csv', default=None)
    ap.add_argument('--tmin', type=float, default=None)
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--flip-sim-cl', action='store_true')
    ap.add_argument('--flip-ref-cl', action='store_true')
    ap.add_argument('--show', action='store_true')
    args=ap.parse_args()

    st,scd,scl=read_sim(args.sim_csv)
    if args.flip_sim_cl: scl=[-x for x in scl]
    st,scd,scl=crop(st,scd,scl,args.tmin,args.tmax)
    print('\n=== Anabasis simulation ===')
    stats('CD_vector', st, scd); stats('CL_z_vector', st, scl)

    bench=None
    if args.bench:
        bt,bcd,bcl=read_bench(args.bench, 'Lift')
        if args.flip_ref_cl: bcl=[-x for x in bcl]
        bt,bcd,bcl=crop(bt,bcd,bcl,args.tmin,args.tmax)
        bench=(bt,bcd,bcl)
        print('\n=== Benchmark: Drag and Lift->CL_z ===')
        stats('Benchmark Drag', bt, bcd); stats('Benchmark Lift mapped to CL_z', bt, bcl)

    if args.clean_csv:
        out=Path(args.clean_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            w=csv.writer(f); w.writerow(['time','CD_vector','CL_z_vector'])
            for row in zip(st,scd,scl): w.writerow([f'{row[0]:.12e}',f'{row[1]:.12e}',f'{row[2]:.12e}'])
        print(f'\nWrote extracted CSV: {out}')

    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
    ax[0].plot(st,scd,label='Anabasis CD_vector'); ax[0].set_ylabel('C_D'); ax[0].grid(True); ax[0].legend()
    ax[1].plot(st,scl,label='Anabasis CL_z_vector'); ax[1].set_ylabel('C_L,z'); ax[1].set_xlabel('time'); ax[1].grid(True); ax[1].legend()
    if bench:
        bt,bcd,bcl=bench
        ax[0].plot(bt,bcd,'--',label='Benchmark Drag'); ax[0].legend()
        ax[1].plot(bt,bcl,'--',label='Benchmark Lift -> CL_z'); ax[1].legend()
    fig.tight_layout()
    out=Path(args.out) if args.out else Path(args.sim_csv).with_name('cd_clz_plot.png')
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out,dpi=200)
    print(f'\nWrote plot: {out}')
    if args.show: plt.show()

if __name__=='__main__': main()
