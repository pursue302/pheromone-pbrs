"""Parameterize the frozen author's learn() trace-reset control."""
from pathlib import Path
import sys,json,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace/research_story_20261120'))
import direct_reset as frozen

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=100)
    p.add_argument('--start',type=int,default=20000);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Frozen results cannot be overwritten')
    m=dict(status='running',seeds=a.seeds,seed_start=a.start,backend='actual_author_learn_with_trace_reset',
        frozen_source_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest())
    mp=out/'manifest.json';mp.write_text(json.dumps(m,indent=2));start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(frozen.one,[(str(out),s) for s in range(a.start,a.start+a.seeds)]))
    m.update(status='complete',steps=sum(r['steps'] for r in rows),seconds=time.perf_counter()-start)
    mp.write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
