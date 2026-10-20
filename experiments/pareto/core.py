"""Numerical and evaluation core for the new Pareto study.

This is a new protocol, not a reproduction of the historical measurements.
No training framework is required to test the generator, decoder or metrics.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PROTOCOL = 'pareto-v2-fixed-particle-truth-2026-10-20'


def uid(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def pareto_mask(time_values, quality_values):
    """Minimize time and maximize quality; equal points are both nondominated."""
    t = np.asarray(time_values, dtype=float)
    q = np.asarray(quality_values, dtype=float)
    valid = np.isfinite(t) & np.isfinite(q)
    result = valid.copy()
    for i in np.flatnonzero(valid):
        dominated = valid & (t <= t[i]) & (q >= q[i]) & ((t < t[i]) | (q > q[i]))
        result[i] = not dominated.any()
    return result


from generator.analytic import advance_z


def simulate_event(seed, n_particles=50, noise_rate=0.4, efficiency=0.99,
                   half_size=0.25, resolution=0.0, n_layers=7):
    """Use the repository generator and detector response, not a notebook copy."""
    from generator.models import ParticleGenerator
    from generator.generation import example_generator_config, build_naive_detectors
    from generator.analytic import simulate_hits_analytic
    if resolution != 0:
        raise ValueError("NaiveDetector has no coordinate-smearing model")
    config = example_generator_config()
    config['n_particles'] = n_particles
    event = ParticleGenerator(config, seed=seed).generate_event()
    setup = build_naive_detectors(n_layers=n_layers, seed=seed)
    for detector in setup.detectors:
        detector.noise_rate = noise_rate
        detector.efficiency = efficiency
        detector.size_x = detector.size_y = 2 * half_size
    hits = simulate_hits_analytic(event, setup, event_id=int(seed))
    np.random.default_rng(seed).shuffle(hits)
    return {'xyz': np.asarray([[h.x,h.y,h.z] for h in hits], dtype=np.float32).reshape(-1,3),
            'layer': np.asarray([h.layer for h in hits], dtype=np.int64),
            'pid': np.asarray([-1 if h.is_fake else h.true_particle_id for h in hits], dtype=np.int64),
            'event_id': int(seed), 'n_generated': int(n_particles)}


def load_legacy_dataset(path):
    """Import supplied dataset.json without regenerating its hits or coordinates."""
    source = json.loads(Path(path).read_text())
    events = []
    for index, hits in enumerate(source['hits']):
        events.append({'xyz': np.asarray([[h['x'],h['y'],h['z']] for h in hits],
                                        dtype=np.float32).reshape(-1,3),
                       'layer': np.asarray([h['layer'] for h in hits],dtype=np.int64),
                       'pid': np.asarray([-1 if h.get('is_fake') or
                                         h.get('true_particle_id') is None
                                         else h['true_particle_id'] for h in hits],dtype=np.int64),
                       'event_id': index,
                       'n_generated': len(source.get('events',[{}]*len(source['hits']))[index].get('particles',[]))})
    return events


def truth_tracks(event, min_hits=3):
    # Frozen before filtering; positions are ORIGINAL array indices in the event.
    return [set(np.flatnonzero(event['pid'] == p).tolist())
            for p in np.unique(event['pid']) if p >= 0
            and np.count_nonzero(event['pid'] == p) >= min_hits]


def graph_arrays(event, k=1):
    xyz = event['xyz']
    if len(xyz):
        # Same event-wise normalization rule for raw and filtered processing.
        x = (xyz - xyz.mean(0)) / (xyz.std(0, ddof=1) + 1e-8) if len(xyz)>1 else xyz*0
    else:
        x = xyz.copy()
    src, dst = [], []
    # Numerical layer gaps are respected, rather than skipped by occupied-layer order.
    for l in np.unique(event['layer']):
        left = np.flatnonzero(event['layer'] == l)
        for gap in range(1, k+1):
            right = np.flatnonzero(event['layer'] == l+gap)
            src.extend(np.repeat(left, len(right))); dst.extend(np.tile(right, len(left)))
    edges = np.asarray([src, dst], dtype=np.int64).reshape(2,-1)
    pid = event['pid']
    y = ((pid[edges[0]] >= 0) & (pid[edges[0]] == pid[edges[1]])).astype(np.float32)
    return np.nan_to_num(x).astype(np.float32), edges, y


def decode_paths(edges, probabilities, node_ids, threshold=.15, beam_width=1,
                 min_hits=3):
    """Forward greedy (B=1) or beam (B>1), with identical conflict rules.

    Rank by length first, then mean log probability; state this explicitly in paper.
    One shared used-hit set ensures vertex-disjoint paths for BOTH algorithms.
    This is a NEW, harmonized decoder, not the exact historical beam implementation.
    """
    adj, indegree = {}, {}
    for (u,v), p in zip(np.asarray(edges).T, probabilities):
        if p > threshold:
            u,v = int(u),int(v)
            adj.setdefault(u,[]).append((v,float(p)))
            indegree[v] = indegree.get(v,0)+1
            indegree.setdefault(u,0)
    for u in adj:
        adj[u].sort(key=lambda t:(-t[1],t[0]))
    used, tracks = set(), []
    starts = sorted(u for u in indegree if indegree[u] == 0)
    # Internal starts are considered later if their predecessors were consumed.
    starts += sorted(u for u in adj if u not in starts)
    def rank(item):
        path, score = item
        return (len(path), score/max(len(path)-1,1), tuple(-v for v in path))
    for start in starts:
        if start in used:
            continue
        beam = [([start], 0.)]
        while True:
            expanded, changed = [], False
            for path, score in beam:
                choices = [(v,p) for v,p in adj.get(path[-1],[]) if v not in used and v not in path]
                if choices:
                    changed = True
                    expanded.extend((path+[v], score+np.log(max(p,1e-12))) for v,p in choices)
                else:
                    expanded.append((path,score))
            beam = sorted(expanded,key=rank,reverse=True)[:beam_width]
            if not changed:
                break
        path = max(beam,key=rank)[0]
        if len(path) >= min_hits:
            used.update(path)
            tracks.append([int(node_ids[v]) for v in path])
    return tracks


def track_counts(predicted, reference, purity=.8, completeness=.8):
    predicted = [set(p) for p in predicted]
    if not predicted or not reference:
        return {'tp':0,'n_pred':len(predicted),'n_true':len(reference),'duplicates':0}
    overlap = np.asarray([[len(p&t) for t in reference] for p in predicted],dtype=float)
    pp = overlap / np.asarray([len(p) for p in predicted])[:,None]
    rr = overlap / np.asarray([len(t) for t in reference])[None,:]
    good = (pp>=purity)&(rr>=completeness)
    # Maximum CARDINALITY among admissible matches, then overlap as tie-breaker.
    row,col = linear_sum_assignment(-(good.astype(float)+.0001*overlap/max(overlap.max(),1)))
    matched = good[row,col]
    tp = int(matched.sum())
    assigned_pred = set(row[matched].tolist())
    duplicates = sum(bool(good[i].any()) for i in range(len(predicted)) if i not in assigned_pred)
    return {'tp':tp,'n_pred':len(predicted),'n_true':len(reference),'duplicates':duplicates}


def summarize_counts(rows):
    sums = {k:sum(r[k] for r in rows) for k in ['tp','n_pred','n_true','duplicates']}
    p=sums['tp']/max(sums['n_pred'],1);r=sums['tp']/max(sums['n_true'],1)
    return {**sums,'precision':p,'recall':r,'f1':2*p*r/max(p+r,1e-12)}


def select_budget_points(frame, maximum=5):
    f=frame.loc[pareto_mask(frame.time_ms,frame.f1)].sort_values('time_ms')
    chosen=[]
    if len(f):
        for budget in np.linspace(f.time_ms.min(),f.time_ms.max(),maximum):
            within=f[f.time_ms<=budget+1e-12]
            chosen.append(within.sort_values(['f1','time_ms'],ascending=[False,True]).iloc[0].config_id)
    return list(dict.fromkeys(chosen))
