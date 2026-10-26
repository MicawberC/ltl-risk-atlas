"""Reuse the frozen timing/evaluation driver with an explicit new controller."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import optimized_controller as opt
old=opt.old

def main():
    method=sys.argv[sys.argv.index('--method')+1]
    if method in ('optimized','priced'):
        warm=method=='optimized'
        old.Controller=lambda m,d,t,_:opt.OptimizedController(m,d,t,warm=warm)
        old.METHODS=(*old.METHODS,method)
    old.main()
if __name__=='__main__':main()
