from .core import *
from .study import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_plot(fig,path):
    path=Path(path)
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.show()


def plot_frontier(state):
    s=state['summary'];folder=state['session']
    tags=s.filter_tag.unique();fig,axes=plt.subplots(1,len(tags),figsize=(5*len(tags),4),squeeze=False,sharey=True)
    colors={1:'#2878b5',2:'#e87524',3:'#e87524',4:'#e87524'}
    markers={'gcn':'o','sage':'s','gat':'^','edgeconv':'D'}
    front=s[s.pareto].sort_values('time_ms')
    for ax,tag in zip(axes[0],tags):
        sub=s[s.filter_tag==tag]
        for _,r in sub.iterrows():
            ax.scatter(r.time_ms,r.f1,color=colors[r.beam_width],marker=markers[r.architecture],
                       alpha=1 if r.pareto else .22,s=25+r.hidden/2,
                       edgecolors='black' if r.pareto else 'none',linewidths=.5)
        ax.step(front.time_ms,front.f1,where='post',c='black',lw=1.2)
        ax.set(title=f'Фильтр / Filter: {tag}',xlabel='Полное время / Full latency, ms')
        ax.grid(alpha=.2);ax.set_ylim(0,1)
    axes[0,0].set_ylabel('Track F1 / F1 треков')
    handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for l,c in [('Greedy','#2878b5'),('Beam','#e87524')]]
    handles += [Line2D([],[],marker=m,ls='',color='black',label=a) for a,m in markers.items()]
    fig.legend(handles=handles,loc='upper center',ncol=6,bbox_to_anchor=(.5,1.08))
    fig.suptitle('Validation sweep — all configurations / Валидационный перебор',y=1.16)
    save_plot(fig,folder/'01_validation_pareto')
    fig,ax=plt.subplots(figsize=(7,4))
    for tag in tags:
        sub=s[s.filter_tag==tag]
        f=sub.loc[pareto_mask(sub.time_ms,sub.f1)].sort_values('time_ms')
        ax.step(f.time_ms,f.f1,where='post',marker='o',label=f'Filter {tag}')
    ax.set(xlabel='Time budget / Бюджет времени, ms',ylabel='Best validation F1 / Лучший F1')
    ax.legend();ax.grid(alpha=.2)
    save_plot(fig,folder/'02_budget_quality')
    fig,ax=plt.subplots(figsize=(6,4))
    for architecture,sub in s.groupby('architecture'):
        ax.scatter(sub.edge_f1,sub.f1,label=architecture,alpha=.6)
    ax.set(xlabel='Edge F1 / F1 рёбер',ylabel='Track F1 / F1 треков');ax.legend();ax.grid(alpha=.2)
    save_plot(fig,folder/'03_edge_vs_track')
    # Fix model, filter, graph and beam: never connect points from different configurations.
    best=s.sort_values('f1',ascending=False).iloc[0]
    sub=s[(s.model_id==best.model_id)&(s.filter_tag==best.filter_tag)&(s.k==best.k)]
    fig,ax=plt.subplots(figsize=(6,4))
    for beam,g in sub.groupby('beam_width'):
        g=g.sort_values('threshold');ax.plot(g.threshold,g.f1,'o-',label=f'B={beam}')
    ax.set(xlabel='Probability threshold / Порог вероятности',ylabel='Track F1')
    ax.legend();ax.grid(alpha=.2)
    save_plot(fig,folder/'04_threshold_sensitivity')


def bootstrap_test(state,repetitions=400):
    """Hierarchical paired resampling of training seeds and test events.

    Intervals are exploratory, especially with only one or three training seeds.
    They characterize quality, not hardware latency uncertainty.
    """
    details=pd.read_csv(state['session']/'test_events.csv')
    seeds=sorted(details.seed.unique());events=sorted(details.event_id.unique())
    configs=state['selected'];rng=np.random.default_rng(771)
    arrays={}
    for cid in configs:
        arrays[cid]={seed:details[(details.config_id==cid)&(details.seed==seed)]
                    .set_index('event_id').loc[events,['tp','n_pred','n_true']].to_numpy()
                    for seed in seeds}
    samples={cid:[] for cid in configs}
    for _ in range(repetitions):
        drawn_seeds=rng.choice(seeds,len(seeds),replace=True)
        event_indices=rng.integers(0,len(events),len(events))
        for cid in configs:
            fs=[]
            for seed in drawn_seeds:
                tp,npred,ntrue=arrays[cid][seed][event_indices].sum(0)
                fs.append(2*tp/max(npred+ntrue,1))
            samples[cid].append(np.mean(fs))
    rows=[]
    for cid in configs:
        values=[summarize_counts(details[(details.config_id==cid)&(details.seed==seed)].to_dict('records'))['f1']
                for seed in seeds]
        rows.append({'config_id':cid,'test_f1_mean':np.mean(values),
                     'test_f1_std':np.std(values,ddof=1) if len(values)>1 else np.nan,
                     'ci_low':np.percentile(samples[cid],2.5),'ci_high':np.percentile(samples[cid],97.5),
                     'n_seeds':len(seeds),'n_test_events':len(events)})
    result=pd.DataFrame(rows)
    result.to_csv(state['session']/'test_quality_intervals.csv',index=False)
    measured=pd.read_csv(state['session']/'test_selected.csv').groupby('config_id').time_ms.mean()
    fig,ax=plt.subplots(figsize=(7,4))
    for _,r in result.iterrows():
        ax.errorbar(measured[r.config_id],r.test_f1_mean,
                    yerr=[[max(0,r.test_f1_mean-r.ci_low)],[max(0,r.ci_high-r.test_f1_mean)]],
                    fmt='o',capsize=4,label=r.config_id[:6])
    ax.set(xlabel='Full test latency / Полное время на тесте, ms',ylabel='Test track F1',
           title='Validation-selected configurations / Выбор по валидации')
    ax.grid(alpha=.2);ax.legend(title='Config ID')
    save_plot(fig,state['session']/'05_test_selected')
    return result


def robustness_check(state,n_events=60):
    """Optional stress test, with frozen configurations and weights; no retuning.

    Applicable to synthetic-v2 generation only. Results from a legacy dataset
    cannot be interpreted as same-domain robustness without a matching generator.
    """
    manifest=json.loads((state['session']/'manifest.json').read_text())
    if manifest['dataset']['mode']!='corrected-synthetic-v2':
        raise ValueError('Robustness generator differs from imported legacy data; run a synthetic-v2 study first.')
    rows=[]
    for seed in state['settings']['seeds']:
        for cid in state['selected']:
            cfg=state['registry'][cid];mid=uid(cfg['model'])
            edge=TrackNet(**cfg['model']).to(DEVICE)
            edge.load_state_dict(torch.load(state['runroot']/f'edge_{mid}_seed{seed}.pt',map_location=DEVICE,weights_only=True));edge.eval()
            hit=None
            if cfg['filter_threshold'] is not None:
                hit=TrackNet(architecture='edgeconv',hidden=32,node_task=True).to(DEVICE)
                hit.load_state_dict(torch.load(state['runroot']/f'hit_seed{seed}.pt',map_location=DEVICE,weights_only=True));hit.eval()
            # Two one-factor sweeps rather than a costly full Cartesian grid.
            scenarios=[('multiplicity',n,.4) for n in [25,50,100]]+[
                       ('noise',50,r) for r in [0.,.4,.8]]
            for dimension,n,noise in scenarios:
                events=[simulate_event(900000+i,n_particles=n,noise_rate=noise) for i in range(n_events)]
                cache=[predict_event(e,edge,hit,cfg) for e in events]
                quality,_=quality_from_cache(events,cache,cfg)
                times=benchmark(events,edge,hit,cfg,state['settings'])
                rows.append({'config_id':cid,'seed':seed,'dimension':dimension,
                             'n_particles':n,'noise_rate':noise,**quality,
                             'time_ms':np.median([r['time_ms'] for r in times])})
            pd.DataFrame(rows).to_csv(state['session']/'robustness.csv',index=False)
    df=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,dimension,x in zip(axes,['multiplicity','noise'],['n_particles','noise_rate']):
        for cid,g in df[df.dimension==dimension].groupby('config_id'):
            m=g.groupby(x).f1.mean().sort_index();ax.plot(m.index,m,'o-',label=cid[:6])
        ax.set(xlabel=x,ylabel='Track F1');ax.grid(alpha=.2);ax.legend()
    save_plot(fig,state['session']/'06_robustness')
    return df
