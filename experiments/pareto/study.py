from .core import *
"""PyTorch/PyG runner for a separate, corrected Pareto study."""
import copy
import csv
import gzip
import os
import platform
import random
import time
import uuid
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv, EdgeConv, GATv2Conv,
                                TransformerConv, GINEConv, ResGatedGraphConv)
from importlib.metadata import version

CODE_HASH = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(
    Path(__file__).resolve().parents[2].rglob('*.py')))).hexdigest()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRESETS = {
    'smoke': dict(n_train=60,n_val=12,n_test=12,epochs=3,seeds=[0],
                  architectures=[('sage',32),('edgeconv',32)],
                  thresholds=[.05,.15,.5],beams=[1,2],filters=[None,.5],
                  k_values=[1],timing_events=6,timing_repeats=1,warmup=2),
    'screen': dict(n_train=600,n_val=120,n_test=200,epochs=30,seeds=[0],
                   architectures=[('gcn',64),('sage',64),('gat',32),('edgeconv',64)],
                   thresholds=[.01,.05,.10,.15,.20,.35,.50],beams=[1,2,3],
                   filters=[None,.1,.5],k_values=[1],timing_events=12,
                   timing_repeats=2,warmup=3),
    'paper': dict(n_train=900,n_val=200,n_test=300,epochs=60,seeds=[0,1,2],
                  architectures=[(a,h) for a in ['gcn','sage','gat','edgeconv']
                                 for h in [32,64]],
                  thresholds=[.01,.05,.10,.15,.20,.35,.50],beams=[1,2,3,4],
                  filters=[None,.1,.5,.8],k_values=[1],timing_events=30,
                  timing_repeats=5,warmup=5),
}


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    # GPU scatter reductions can still be nondeterministic. Save seeds AND report repeats.
    torch.backends.cudnn.benchmark=False


def sync():
    if DEVICE.type=='cuda':torch.cuda.synchronize(DEVICE)


def as_graph(event,k=1):
    x,edges,y=graph_arrays(event,k)
    return Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges),
                edge_label=torch.from_numpy(y),
                node_label=torch.from_numpy((event['pid']>=0).astype(np.float32)),
                original_id=torch.arange(len(x),dtype=torch.long))


class TrackNet(nn.Module):
    """Edge or hit classifier with interchangeable message-passing operators.

    Existing defaults and parameter names remain unchanged for checkpoint
    compatibility. Geometric edge attributes are signed normalized coordinate
    differences (destination minus source).
    """
    EDGE_AWARE = frozenset({'gine', 'transformer'})

    def __init__(self, architecture='edgeconv', hidden=64, layers=3,
                 node_task=False, heads=2, dropout=0.1):
        super().__init__()
        if hidden < 1 or layers < 1 or heads < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid hidden/layers/heads/dropout')
        self.node_task = node_task
        self.architecture = architecture
        self.dropout = dropout
        self.input_proj = nn.Linear(3, hidden)

        def conv():
            if architecture == 'gcn':
                return GCNConv(hidden, hidden)
            if architecture == 'sage':
                return SAGEConv(hidden, hidden)
            if architecture == 'gat':
                return GATConv(hidden, hidden, heads=heads, concat=False)
            if architecture == 'gatv2':
                return GATv2Conv(hidden, hidden, heads=heads, concat=False)
            if architecture == 'transformer':
                return TransformerConv(hidden, hidden, heads=heads,
                                       concat=False, edge_dim=3)
            if architecture == 'gine':
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden))
                return GINEConv(mlp, edge_dim=3)
            if architecture == 'resgated':
                return ResGatedGraphConv(hidden, hidden)
            if architecture == 'edgeconv':
                # Independent MLP per layer, preserving the previous behavior.
                return EdgeConv(nn.Sequential(nn.Linear(2 * hidden, hidden),
                                              nn.ReLU(), nn.Linear(hidden, hidden)))
            raise ValueError(f'Unknown architecture: {architecture}')

        self.convs = nn.ModuleList([conv() for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden if node_task else 2 * hidden, 64),
            nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 32),
            nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, data):
        x = self.input_proj(data.x)
        edge_attr = None
        if self.architecture in self.EDGE_AWARE:
            src, dst = data.edge_index
            edge_attr = data.x[dst] - data.x[src]
        for conv in self.convs:
            if edge_attr is None:
                updated = conv(x, data.edge_index)
            else:
                updated = conv(x, data.edge_index, edge_attr=edge_attr)
            x = x + F.dropout(F.relu(updated), p=self.dropout, training=self.training)
        features = x if self.node_task else torch.cat(
            [x[data.edge_index[0]], x[data.edge_index[1]]], dim=1)
        return self.head(features).flatten()


def prune_graph(graph,keep):
    indices=torch.where(keep)[0]
    mapping=torch.full((graph.num_nodes,),-1,dtype=torch.long,device=graph.x.device)
    mapping[indices]=torch.arange(len(indices),device=graph.x.device)
    src,dst=graph.edge_index
    ekeep=keep[src]&keep[dst]
    return Data(x=graph.x[indices],edge_index=mapping[graph.edge_index[:,ekeep]],
                edge_label=graph.edge_label[ekeep],node_label=graph.node_label[indices],
                original_id=graph.original_id[indices])


def get_dataset(settings,legacy_path=None):
    n=settings['n_train']+settings['n_val']+settings['n_test']
    if legacy_path:
        events=load_legacy_dataset(legacy_path)
        if len(events)<n:
            raise ValueError(f'Dataset has {len(events)} events; preset needs {n}. Reduce split sizes explicitly.')
        source={'mode':'legacy-json','sha256':hashlib.sha256(Path(legacy_path).read_bytes()).hexdigest()}
    else:
        events=[simulate_event(100000+i) for i in range(n)]
        source={'mode':'corrected-synthetic-v2','n':n,'first_seed':100000,
                'noise_rate':.4,'n_particles':50,'normalization':'event-wise ddof=1',
                'field':'analytic Bz helix','noise_acceptance':'same as true hits'}
    order=np.random.default_rng(42).permutation(len(events))[:n]
    a=settings['n_train'];b=a+settings['n_val']
    return {'train':[events[i] for i in order[:a]],
            'val':[events[i] for i in order[a:b]],
            'test':[events[i] for i in order[b:]]},source


def train_net(model_cfg,seed,train_events,val_events,epochs,path,node_task=False):
    set_seed(seed)
    net=TrackNet(**model_cfg,node_task=node_task).to(DEVICE)
    if path.exists():
        net.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));net.eval()
        return net
    graphs=[as_graph(e,1) for e in train_events]
    val_graphs=[as_graph(e,1) for e in val_events]
    optimizer=torch.optim.Adam(net.parameters(),lr=.005)
    best,best_loss,history=None,float('inf'),[]
    for epoch in range(epochs):
        net.train();loss_sum=0.;objects=0
        # The order is reproducible for this training seed.
        for i in np.random.permutation(len(graphs)):
            g=graphs[i].to(DEVICE)
            if (g.num_nodes if node_task else g.num_edges)==0:continue
            optimizer.zero_grad();logits=net(g)
            y=g.node_label if node_task else g.edge_label
            loss=F.binary_cross_entropy_with_logits(logits,y)
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(y);objects+=len(y)
        net.eval();vsum=0.;vn=0
        with torch.inference_mode():
            for graph in val_graphs:
                g=graph.to(DEVICE)
                if (g.num_nodes if node_task else g.num_edges)==0:continue
                y=g.node_label if node_task else g.edge_label
                vsum+=float(F.binary_cross_entropy_with_logits(net(g),y,reduction='sum'));vn+=len(y)
        vl=vsum/max(vn,1)
        history.append({'epoch':epoch+1,'train_loss':loss_sum/max(objects,1),'val_loss':vl})
        if vl<best_loss:
            best_loss=vl;best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            print(f'{path.stem}: {epoch+1}/{epochs}; train={history[-1]["train_loss"]:.4f}, val={vl:.4f}',flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(best,path);pd.DataFrame(history).to_csv(path.with_suffix('.loss.csv'),index=False)
    net.load_state_dict(best);net.eval();return net


@torch.inference_mode()
def predict_event(event,edge_model,hit_model,config):
    g=as_graph(event,config['k']).to(DEVICE)
    if config['filter_threshold'] is not None and g.num_nodes:
        keep=torch.sigmoid(hit_model(g))>=config['filter_threshold']
        g=prune_graph(g,keep)
    if g.num_edges:
        probs=torch.sigmoid(edge_model(g)).cpu().numpy()
    else:probs=np.empty(0,dtype=float)
    return (g.edge_index.cpu().numpy(),probs,g.original_id.cpu().numpy(),
            g.edge_label.cpu().numpy(),int(g.num_nodes))


def process_event(event,edge_model,hit_model,config):
    # Full hit-to-candidate path: graph, H2D, optional hit network/pruning,
    # edge network, sigmoid, D2H, decoder. Does not include detector simulation or scoring.
    edges,probs,ids,labels,n_nodes=predict_event(event,edge_model,hit_model,config)
    tracks=decode_paths(edges,probs,ids,config['threshold'],config['beam_width'])
    return tracks


def benchmark(events,edge_model,hit_model,config,settings):
    subset=events[:settings['timing_events']]
    for i in range(settings['warmup']):
        process_event(subset[i%len(subset)],edge_model,hit_model,config)
    samples=[]
    # Keep repeats nested within event for paired bootstrap and event-size diagnostics.
    for event in subset:
        for repeat in range(settings['timing_repeats']):
            sync();start=time.perf_counter_ns()
            process_event(event,edge_model,hit_model,config)
            sync();duration=(time.perf_counter_ns()-start)/1e6
            samples.append({'event_id':event['event_id'],'repeat':repeat,'time_ms':duration})
    return samples


def quality_from_cache(events,predictions,config):
    rows=[];edge_tp=edge_fp=edge_fn=0
    for event,(edges,probs,ids,labels,n_nodes) in zip(events,predictions):
        tracks=decode_paths(edges,probs,ids,config['threshold'],config['beam_width'])
        counts=track_counts(tracks,truth_tracks(event))
        rows.append({**counts,'event_id':event['event_id'],'n_hits':len(event['xyz']),
                     'retained_hits':n_nodes,'n_edges':len(probs)})
        binary=probs>config['threshold'];positive=labels.astype(bool)
        edge_tp+=int((binary&positive).sum());edge_fp+=int((binary&~positive).sum());edge_fn+=int((~binary&positive).sum())
    ep=edge_tp/max(edge_tp+edge_fp,1);er=edge_tp/max(edge_tp+edge_fn,1)
    return {**summarize_counts(rows),'edge_precision':ep,'edge_recall':er,
            'edge_f1':2*ep*er/max(ep+er,1e-12)},rows


def aggregate_validation(frame):
    keys=['config_id','model_id','architecture','hidden','k','threshold','beam_width','filter_tag']
    agg=frame.groupby(keys,dropna=False).agg(f1=('f1','mean'),f1_std=('f1','std'),
        time_ms=('time_ms','mean'),time_std=('time_ms','std'),precision=('precision','mean'),
        recall=('recall','mean'),edge_f1=('edge_f1','mean')).reset_index()
    agg['pareto']=pareto_mask(agg.time_ms,agg.f1)
    return agg


def run_validation(profile='smoke',output_root='pareto_runs',legacy_path=None):
    settings=copy.deepcopy(PRESETS[profile])
    if settings['k_values'] != [1]:
        raise ValueError('This protocol trains at k=1; use k_values=[1].')
    data,source=get_dataset(settings,legacy_path)
    specification={'protocol':PROTOCOL,'profile':profile,'settings':settings,'dataset':source,
                   'code_hash':globals().get('CODE_HASH','unspecified')}
    runroot=Path(output_root)/uid(specification);runroot.mkdir(parents=True,exist_ok=True)
    session=runroot/'sessions'/time.strftime('%Y%m%d_%H%M%S')
    session.mkdir(parents=True,exist_ok=True)
    metadata={**specification,'python':platform.python_version(),'torch':torch.__version__,
              'pyg':version('torch-geometric'),'numpy':np.__version__,'device':str(DEVICE),
              'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
              'cpu':platform.processor(),'torch_threads':torch.get_num_threads(),
              'session_id':str(uuid.uuid4()),'repository_commit':os.environ.get('FIXED_TARGET_COMMIT'),'decoder':'forward disjoint length-first mean-log beam',
              'primary_metric':'micro track F1 per training seed; mean across seeds',
              'time_boundary':'CPU raw hit arrays to CPU disjoint track candidates; synchronized wall time'}
    (session/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    np.savez_compressed(runroot/'split_ids.npz',**{k:np.array([e['event_id'] for e in v]) for k,v in data.items()})
    with gzip.open(runroot/'dataset.json.gz','wt') as f:
        json.dump({k:[{a:(b.tolist() if isinstance(b,np.ndarray) else b) for a,b in e.items()} for e in v]
                   for k,v in data.items()},f)
    registry={};rows=[];timing_rows=[]
    for seed in settings['seeds']:
        hit=None
        if any(x is not None for x in settings['filters']):
            hit=train_net(dict(architecture='edgeconv',hidden=32),seed,data['train'],data['val'],
                          settings['epochs'],runroot/f'hit_seed{seed}.pt',node_task=True)
        for architecture,hidden in settings['architectures']:
            model_cfg=dict(architecture=architecture,hidden=hidden)
            mid=uid(model_cfg)
            edge=train_net(model_cfg,seed,data['train'],data['val'],settings['epochs'],
                           runroot/f'edge_{mid}_seed{seed}.pt')
            for k in settings['k_values']:
                for ft in settings['filters']:
                    base={'model':model_cfg,'k':k,'filter_threshold':ft}
                    # One inference pass per event for quality scanning. Timing below NEVER uses cache.
                    cached=[predict_event(e,edge,hit,base) for e in data['val']]
                    jobs=[{**base,'threshold':float(th),'beam_width':int(b)}
                          for th in settings['thresholds'] for b in settings['beams']]
                    # Avoid a fixed ordered sequence of decoder/timing configurations.
                    random.Random(seed+919).shuffle(jobs)
                    for config in jobs:
                        cid=uid(config);registry[cid]=config
                        quality,_=quality_from_cache(data['val'],cached,config)
                        times=benchmark(data['val'],edge,hit,config,settings)
                        ms=float(np.median([t['time_ms'] for t in times]))
                        rows.append({'config_id':cid,'model_id':mid,'seed':seed,
                            'architecture':architecture,'hidden':hidden,'k':k,
                            'threshold':config['threshold'],'beam_width':config['beam_width'],
                            'filter_tag':'off' if ft is None else str(ft),'time_ms':ms,
                            'time_p95_ms':float(np.percentile([t['time_ms'] for t in times],95)),
                            **quality})
                        timing_rows.extend({**t,'config_id':cid,'seed':seed} for t in times)
                    pd.DataFrame(rows).to_csv(session/'validation_all.csv',index=False)
                    pd.DataFrame(timing_rows).to_csv(session/'validation_timing.csv',index=False)
                    (session/'configurations.json').write_text(json.dumps(registry,indent=2))
                    print('Finished',seed,architecture,hidden,'k',k,'filter',ft,flush=True)
            del edge
            if DEVICE.type=='cuda':torch.cuda.empty_cache()
    summary=aggregate_validation(pd.DataFrame(rows))
    summary.to_csv(session/'validation_summary.csv',index=False)
    selected=select_budget_points(summary)
    (session/'selected_before_test.json').write_text(json.dumps(selected,indent=2))
    return dict(settings=settings,data=data,registry=registry,runroot=runroot,session=session,
                validation=pd.DataFrame(rows),summary=summary,selected=selected)


def run_test(state):
    settings=state['settings'];session=state['session'];root=state['runroot']
    # Read the frozen validation selection; NEVER choose a threshold on test quality.
    selected=json.loads((session/'selected_before_test.json').read_text())
    rows=[];event_rows=[];timing_rows=[]
    for seed in settings['seeds']:
        hit=None
        if any(state['registry'][cid]['filter_threshold'] is not None for cid in selected):
            hit=TrackNet(architecture='edgeconv',hidden=32,node_task=True).to(DEVICE)
            hit.load_state_dict(torch.load(root/f'hit_seed{seed}.pt',map_location=DEVICE,weights_only=True));hit.eval()
        for cid in selected:
            config=state['registry'][cid];mid=uid(config['model'])
            edge=TrackNet(**config['model']).to(DEVICE)
            edge.load_state_dict(torch.load(root/f'edge_{mid}_seed{seed}.pt',map_location=DEVICE,weights_only=True));edge.eval()
            cache=[predict_event(e,edge,hit,config) for e in state['data']['test']]
            quality,details=quality_from_cache(state['data']['test'],cache,config)
            times=benchmark(state['data']['test'],edge,hit,config,settings)
            rows.append({'config_id':cid,'seed':seed,**quality,
                         'time_ms':float(np.median([t['time_ms'] for t in times])),
                         'time_p95_ms':float(np.percentile([t['time_ms'] for t in times],95))})
            event_rows.extend({**r,'config_id':cid,'seed':seed} for r in details)
            timing_rows.extend({**r,'config_id':cid,'seed':seed} for r in times)
            del edge
    result=pd.DataFrame(rows)
    result.to_csv(session/'test_selected.csv',index=False)
    pd.DataFrame(event_rows).to_csv(session/'test_events.csv',index=False)
    pd.DataFrame(timing_rows).to_csv(session/'test_timing.csv',index=False)
    return result
