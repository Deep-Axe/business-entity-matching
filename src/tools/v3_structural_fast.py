"""Memoized copy of pure structural definitions, with exact feature parity.

Never mutates the imported canonical module used by running experiments.
Caches are bounded and contain text only. No changed normalization semantics.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np


def definitions():
    name='_v3_structural_memoized_copy'
    if name in sys.modules:return sys.modules[name]
    path=Path(__file__).with_name('v3_structural_features.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    for function in ('folded','core_name','leet_view','state_name','address_view','house_number','street_component','city_proxy'):
        setattr(module,function,lru_cache(maxsize=16000)(getattr(module,function)))
    return module


def matrix(rows,counts,fragments):
    mod=definitions()
    @lru_cache(maxsize=60000)
    def view(name,address,country):return mod.View.make(name,address,country,fragments)
    x=np.empty((len(rows),len(mod.NAMES)),np.float32)
    for i,r in enumerate(rows):
        left,right=view(r[5],r[6],r[3]),view(r[7],r[8],r[3])
        n,ncity=counts.get(r[3],left.core,left.city)
        x[i]=mod.pair_features(left,right,n,ncity)
    if not np.isfinite(x).all():raise ValueError('Invalid memoized feature matrix')
    return x
