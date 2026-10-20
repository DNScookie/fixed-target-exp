from pathlib import Path
import numpy as np

from experiments.pareto.core import *

# Exact dominance, including equal times, duplicates, dominated and invalid points.
assert pareto_mask([1,2,3,2,4,np.nan],[.3,.5,.4,.5,.8,.9]).tolist()==[True,True,False,True,True,False]
assert pareto_mask([1,1,2],[.2,.3,.3]).tolist()==[False,True,False]

# Fragmentation and duplicate candidates cannot inflate reconstructed particle count.
truth=[set(range(7))]
assert track_counts([list(range(7)),list(range(7))],truth)==dict(tp=1,n_pred=2,n_true=1,duplicates=1)
assert track_counts([list(range(3)),list(range(3,7))],truth)['tp']==0
assert track_counts([],truth)['n_true']==1
assert track_counts([[0,1,2]],[{0,1,2}])['tp']==1

# Analytic motion: no-field limit, conserved momentum, inverse-charge symmetry,
# rotation handedness, and invariance to subdivision.
p=np.array([.2,.1,1.]);origin=np.zeros(3)
pos,q=advance_z(origin,p,1,1.,0.)
assert np.allclose(pos,[.2,.1,1.]) and np.allclose(q,p)
a,pa=advance_z(origin,p,1,1.,.8)
b,pb=advance_z(origin,p,1,.5,.8);b,pb=advance_z(b,pb,1,1.,.8)
assert np.allclose(a,b) and np.allclose(pa,pb)
assert np.isclose(np.linalg.norm(pa),np.linalg.norm(p))
_,rotated=advance_z(origin,[1.,0.,1.],1,.1,.8)
assert rotated[1]<0

event=simulate_event(42)
again=simulate_event(42)
assert np.array_equal(event['xyz'],again['xyz'])
assert np.all(np.abs(event['xyz'][:,:2])<=.250001)
x,e,y=graph_arrays(event)
assert np.all(event['layer'][e[1]]-event['layer'][e[0]]==1)
assert np.all((event['pid'][e[0,y==1]]>=0))
assert len(truth_tracks(event))<=50

# Both decoders must use original IDs and enforce the same disjointness rule.
edges=np.array([[0,0,1,2,3,4],[1,2,3,3,4,5]])
for width in [1,2,3]:
    paths=decode_paths(edges,np.array([.9,.8,.9,.99,.9,.9]),np.arange(6)+10,.1,width)
    flat=[v for path in paths for v in path]
    assert len(flat)==len(set(flat)) and all(v>=10 for v in flat)

print('Core checks passed: Pareto, fixed truth, matching, propagation, graph, decoders.')
