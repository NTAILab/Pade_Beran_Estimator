"""Paired, resumable synthetic parameter-series experiments."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
import csv, hashlib, json, math, os
from pathlib import Path
import tempfile, time
from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler
from . import timing as speed_benchmark
from .approximation_metrics import METRIC_NAMES as APPROXIMATION_METRICS, SURVIVAL_LOG_EPS, fold_approximation_errors
from pbestimator.beran import BeranEstimator, PadeBeranEstimator, TaylorBeranEstimator
from .generators import MODEL_NAMES, GeneratorConfig, make_generator, time_predictors
from .evaluation import administratively_censor_test_outcomes, ibs_reporting_horizon
from pbestimator.metrics import concordance_index, event_time_horizon, integrated_brier_score, negative_rmst_risk
from .evaluation import evaluation_time_grid
from pbestimator.synthetic import SyntheticSurvivalDataset

SERIES_MASTER_SEED, SERIES_FOLDS, N_TRAIN, N_TEST = 17000, 20, 2000, 2000
TIMING_WARMUPS, TIMING_REPEATS, EVALUATION_GRID_SIZE, _CENSOR_STREAM = 2, 7, 50, 20260920
SERIES_SCHEMA_VERSION = 2
TIMING_KINDS = ("predict", "increment", "accumulation")
Parameter = Literal["tau", "censor_rate"]

@dataclass(frozen=True)
class StartingState: tau: float; dim: int|None; sparsity: float|None; k: float
STARTING_STATES = {"friedman1":StartingState(.1,5,None,2.5),"friedman2":StartingState(.03,None,None,2.5),"friedman3":StartingState(.03,None,None,2.5),"strong_feature_interactions":StartingState(.1,5,None,2.5),"sparse_features":StartingState(.3,5,.2,2.5),"nonlinear_dependency":StartingState(.1,5,None,2.5)}

@dataclass(frozen=True)
class ExperimentSeries:
    name: str; dataset: str; parameter: Parameter; values: tuple[float,...]; output_dir: Path
    def __post_init__(self):
        if not self.name or self.dataset not in STARTING_STATES: raise ValueError("unknown dataset or empty series name")
        if self.parameter not in ("tau","censor_rate"): raise ValueError("parameter must be tau or censor_rate")
        values=tuple(self.values)
        if not values or len(values)!=len(set(values)): raise ValueError("values must be non-empty and unique")
        for value in values:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError("values must be finite numeric values")
            if self.parameter=="tau" and value<=0: raise ValueError("tau must be positive")
            if self.parameter=="censor_rate" and not 0<=value<1: raise ValueError("censor_rate must be in [0, 1)")
        object.__setattr__(self,"values",tuple(float(v) for v in values)); object.__setattr__(self,"output_dir",Path(self.output_dir))
@dataclass(frozen=True)
class PreparedSeriesFold: train:SyntheticSurvivalDataset; test:SyntheticSurvivalDataset; X_train:np.ndarray; X_test:np.ndarray
@dataclass(frozen=True)
class SeriesFoldResult:
    """One fold/model observation.  Approximation errors are means over test objects
    relative to Exact Log; ``increment``/``accumulation`` are stage-only timings
    (hazards precomputed), ``predict`` is the unchanged full-inference timing."""
    parameter_value:float; fold_index:int; derived_generator_seed:int; model:str; c_index:float; ibs:float; predict_median_ns:float; predict_mean_ns:float; predict_std_ns:float; predict_samples_ns:tuple[int,...]
    survival_ise:float; survival_iae:float; survival_sup:float; cumulative_hazard_iae:float
    increment_median_ns:float; increment_mean_ns:float; increment_std_ns:float; increment_samples_ns:tuple[int,...]
    accumulation_median_ns:float; accumulation_mean_ns:float; accumulation_std_ns:float; accumulation_samples_ns:tuple[int,...]
@dataclass(frozen=True)
class _TimingProtocol: timing_warmups:int=TIMING_WARMUPS; timing_repeats:int=TIMING_REPEATS
_AGGREGATES=(("c_index","c_index",1.),("ibs","ibs",1.),("predict_time_ms","predict_median_ns",1_000_000.),*((m,m,1.) for m in APPROXIMATION_METRICS),("increment_time_ms","increment_median_ns",1_000_000.),("accumulation_time_ms","accumulation_median_ns",1_000_000.))
SERIES_CSV_HEADER=("tau",*(f"{m}_{label}_{stat}" for m in MODEL_NAMES for label,_,_ in _AGGREGATES for stat in ("mean","std")))
_FOLDS_HEADER=tuple(f.name for f in fields(SeriesFoldResult))
_SAMPLE_FIELDS=tuple(f"{k}_samples_ns" for k in TIMING_KINDS)

def derive_series_seeds(master_seed:int=SERIES_MASTER_SEED,fold_count:int=SERIES_FOLDS)->tuple[int,...]:
    if type(master_seed) is not int or master_seed<0 or type(fold_count) is not int or fold_count<1: raise ValueError("invalid master seed/fold count")
    result=tuple(int(c.generate_state(1,dtype=np.uint64)[0]) for c in np.random.SeedSequence(master_seed).spawn(fold_count))
    if len(result)!=len(set(result)): raise ValueError("duplicate derived generator seed")
    return result
def _source_fingerprint(root: Path) -> str:
    """Hash source paths and bytes, independent of location and generated output."""
    digest = hashlib.sha256(b"pbestimator-source-v1\0")
    files = []
    for relative in ("src/pbestimator", "experiments"):
        sources = list((root / relative).rglob("*.py"))
        if not sources:
            raise RuntimeError(f"missing scientific sources: {relative}")
        files.extend(sources)
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(name + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def implementation_fingerprint() -> str:
    """Identify scientific Python sources independently of version control."""
    return _source_fingerprint(Path(__file__).resolve().parents[2])

def _config(dataset:str)->GeneratorConfig:
    s=STARTING_STATES[dataset]; return GeneratorConfig(dataset,0,s.tau,s.dim,s.k,s.sparsity)
def _paired_censor(data:SyntheticSurvivalDataset,rate:float,seed:int)->SyntheticSurvivalDataset:
    event=np.ones(len(data.event_time),dtype=bool); observed=data.event_time.copy(); rng=np.random.default_rng(np.random.SeedSequence([seed,_CENSOR_STREAM])); indices=rng.permutation(len(event))[:math.floor(len(event)*rate)]; factors=rng.uniform(.05,.95,len(event)); event[indices]=False; observed[indices]*=factors[indices]
    return SyntheticSurvivalDataset(data.X.copy(),data.event_time.copy(),observed,event)
def prepare_series_fold(dataset:str,*,seed:int,train_censor_rate:float)->PreparedSeriesFold:
    if dataset not in STARTING_STATES: raise ValueError(f"unknown dataset: {dataset}")
    if not 0<=train_censor_rate<1: raise ValueError("train_censor_rate must be in [0, 1)")
    generator=make_generator(_config(dataset),seed=seed); latent=generator.generate(N_TRAIN,censor_rate=0.); test=generator.generate(N_TEST,censor_rate=0.); train=_paired_censor(latent,train_censor_rate,seed)
    if not np.all(test.event) or not np.array_equal(test.observed_time,test.event_time): raise ValueError("series test data must be uncensored")
    scaler=StandardScaler().fit(train.X); return PreparedSeriesFold(train,test,scaler.transform(train.X),scaler.transform(test.X))
def time_accumulation_stages(models,hazards:np.ndarray,*,seed:int)->tuple[dict[str,tuple[int,...]],dict[str,tuple[int,...]]]:
    """Time only the model-specific post-hazard stages on one shared, precomputed hazard matrix.

    increment: ``_log_increment(hazards)``; accumulation: increment + cumsum + exp
    (``_survival_from_hazards``).  Kernel logits, risk/event reductions and hazard
    construction happen before the clock; the matrix is passed read-only so no timed
    call can mutate it.  Same warmup/repeat/interleaving policy as full inference.
    """
    if set(models)!=set(MODEL_NAMES): raise ValueError(f"models must contain exactly {MODEL_NAMES!r}")
    shared=hazards.view(); shared.flags.writeable=False; shape=shared.shape
    stages={"increment":{n:(lambda m=models[n]:m._log_increment(shared)) for n in MODEL_NAMES},"accumulation":{n:(lambda m=models[n]:m._survival_from_hazards(shared)) for n in MODEL_NAMES}}
    def check(value,name,stage):
        value=np.asarray(value)
        if value.shape!=shape or np.any(np.isnan(value)): raise ValueError(f"{name} {stage} stage returned an invalid matrix")
    for _ in range(TIMING_WARMUPS):
        for stage,calls in stages.items():
            for name in MODEL_NAMES: check(calls[name](),name,stage)
    samples={stage:{n:[] for n in MODEL_NAMES} for stage in stages}; rng=np.random.default_rng(20260924+seed)
    for _ in range(TIMING_REPEATS):
        for stage,calls in stages.items():
            for name in rng.permutation(MODEL_NAMES):
                elapsed,value=speed_benchmark.timed_call(calls[name])
                if type(elapsed) is not int or elapsed<=0: raise ValueError("timing sample must be a positive integer")
                check(value,name,stage); samples[stage][name].append(elapsed); del value
    return tuple({n:tuple(v) for n,v in samples[stage].items()} for stage in ("increment","accumulation"))
def _timing_summary(samples:Sequence[int])->tuple[float,float,float,tuple[int,...]]: return float(np.median(samples)),float(np.mean(samples)),float(np.std(samples,ddof=1)),tuple(samples)
def evaluate_prepared_fold(data:PreparedSeriesFold,*,tau:float,fold_index:int,seed:int)->tuple[SeriesFoldResult,...]:
    models={n:cls(tau=tau).fit(data.X_train,data.train.observed_time,data.train.event) for n,cls in zip(MODEL_NAMES,(BeranEstimator,TaylorBeranEstimator,PadeBeranEstimator))}
    horizon=ibs_reporting_horizon(data.train.observed_time,data.train.event); tmax=event_time_horizon(data.train.observed_time,data.train.event); test_time,test_event=administratively_censor_test_outcomes(data.test.observed_time,data.test.event,horizon=horizon); grid=evaluation_time_grid(data.train.observed_time,test_time,t_max=horizon,n_times=EVALUATION_GRID_SIZE); metrics={}
    for n,m in models.items(): metrics[n]=(concordance_index(data.test.event,data.test.observed_time,negative_rmst_risk(m,data.X_test,tmax)),integrated_brier_score(data.train.observed_time,data.train.event,test_time,test_event,m.predict_survival_function(data.X_test,grid),grid))
    samples=time_predictors(models,data.X_test,grid,protocol=_TimingProtocol(),seed=seed)
    exact=models["exact_log"]; hazards=exact._hazards_from_logits(exact._kernel_logits(data.X_test))  # common to all three models
    errors=fold_approximation_errors(models,data.X_test,horizon,hazards=hazards); increment,accumulation=time_accumulation_stages(models,hazards,seed=seed); del hazards
    return tuple(SeriesFoldResult(0.,fold_index,seed,n,*metrics[n],*_timing_summary(samples[n]),*(errors[n][k] for k in APPROXIMATION_METRICS),*_timing_summary(increment[n]),*_timing_summary(accumulation[n])) for n in MODEL_NAMES)
def build_series_manifest(series:ExperimentSeries,*,folds:int,sha:str)->dict[str,object]:
    if not sha.startswith("sha256:") or len(sha) != 71 or any(c not in "0123456789abcdef" for c in sha[7:].lower()):
        raise ValueError("implementation identity must be a sha256 source fingerprint")
    if type(folds) is not int or folds<1: raise ValueError("folds must be positive")
    return {"schema_version":SERIES_SCHEMA_VERSION,"implementation_fingerprint":sha,"series":series.name,"dataset":series.dataset,"starting_state":asdict(STARTING_STATES[series.dataset]),"parameter":series.parameter,"values":list(series.values),"n_train":N_TRAIN,"n_test":N_TEST,"train_censor_rate":.3 if series.parameter=="tau" else "varied","test_censor_rate":0.,"master_seed":SERIES_MASTER_SEED,"derived_generator_seeds":list(derive_series_seeds(fold_count=folds)),"folds":folds,"timing_warmups":TIMING_WARMUPS,"timing_repeats":TIMING_REPEATS,"timing_kinds":list(TIMING_KINDS),"approximation_reference":"exact_log","approximation_metrics":list(APPROXIMATION_METRICS),"survival_log_eps":SURVIVAL_LOG_EPS,"approximation_integration":"exact right-continuous step intervals over [0, ibs_reporting_horizon]","models":list(MODEL_NAMES),"pairing":"same derived generator seeds for every parameter value","tau_data":"one generated and train-only-preprocessed fold reused for every tau value","censor_data":"uncensored latent train/test generated before deterministic paired train censoring"}
def _atomic(path:Path,text:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent,text=True)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
def _close(left:float,right:float)->bool: return math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-9)
def _validate_row(row:SeriesFoldResult,manifest:dict[str,object])->None:
    values=manifest["values"]; folds=manifest["folds"]; seeds=manifest["derived_generator_seeds"]
    if row.parameter_value not in values or type(row.fold_index) is not int or not 0<=row.fold_index<folds: raise ValueError("invalid persisted parameter/fold")
    if row.derived_generator_seed != seeds[row.fold_index] or row.model not in MODEL_NAMES: raise ValueError("invalid persisted seed/model")
    if not all(math.isfinite(value) for value in (row.c_index,row.ibs)): raise ValueError("invalid persisted metrics")
    errors=[getattr(row,name) for name in APPROXIMATION_METRICS]
    if not all(isinstance(value,float) and math.isfinite(value) and value>=0 for value in errors): raise ValueError("invalid persisted approximation error")
    if row.model=="exact_log" and any(value!=0.0 for value in errors): raise ValueError("Exact Log self-error must be exactly zero")
    for kind in TIMING_KINDS:
        median_ns,mean_ns,std_ns,samples=(getattr(row,f"{kind}_{s}") for s in ("median_ns","mean_ns","std_ns","samples_ns"))
        if not all(math.isfinite(value) and value>0 for value in (median_ns,mean_ns)) or not math.isfinite(std_ns) or std_ns<0: raise ValueError(f"invalid persisted {kind} timing summary")
        if not isinstance(samples,(tuple,list)) or len(samples)!=manifest["timing_repeats"] or any(type(x) is not int or x<=0 for x in samples): raise ValueError(f"invalid persisted {kind} timing samples")
        if not all((_close(median_ns,float(np.median(samples))),_close(mean_ns,float(np.mean(samples))),_close(std_ns,float(np.std(samples,ddof=1))))): raise ValueError(f"inconsistent persisted {kind} timing summary")
def _validate_rows(rows:Sequence[SeriesFoldResult],manifest:dict[str,object])->None:
    keys=set(); groups={}
    for row in rows:
        _validate_row(row,manifest); key=(row.parameter_value,row.fold_index,row.model)
        if key in keys: raise ValueError("duplicate persisted fold/model result")
        keys.add(key); groups.setdefault((row.parameter_value,row.fold_index),set()).add(row.model)
    if any(models != set(MODEL_NAMES) for models in groups.values()): raise ValueError("incomplete persisted fold/model group")
def _parse_row(record:dict[str,str])->SeriesFoldResult:
    values={}
    for name in _FOLDS_HEADER:
        raw=record[name]
        if raw is None: raise ValueError("missing folds.csv field")
        values[name]=tuple(json.loads(raw)) if name in _SAMPLE_FIELDS else int(raw) if name in ("fold_index","derived_generator_seed") else raw if name=="model" else float(raw)
    return SeriesFoldResult(**values)
class _SeriesStore:
    def __init__(self,root,manifest,rows): self.root,self.manifest,self.rows=root,manifest,rows
    @classmethod
    def open(cls,series,*,manifest,resume):
        root=series.output_dir
        if root.exists() and not resume and any(root.iterdir()): raise ValueError("invalid output collision")
        root.mkdir(parents=True,exist_ok=True); target=root/"manifest.json"
        if target.exists():
            existing=json.loads(target.read_text(encoding="utf-8")); version=existing.get("schema_version") if isinstance(existing,dict) else None
            if version!=manifest["schema_version"]: raise ValueError(f"incompatible resume: series schema version {version!r} != {manifest['schema_version']} (old runs remain historical evidence but cannot be resumed)")
            if json.loads(target.read_text(encoding="utf-8"))!=manifest: raise ValueError("incompatible resume manifest")
        elif resume and any(root.iterdir()): raise ValueError("incompatible resume: missing manifest")
        else: _atomic(target,json.dumps(manifest,sort_keys=True,indent=2)+"\n")
        rows=[]; path=root/"folds.csv"
        if path.exists():
            with path.open(encoding="utf-8",newline="") as f:
                reader=csv.DictReader(f,delimiter=";")
                if tuple(reader.fieldnames or ())!=_FOLDS_HEADER: raise ValueError("invalid folds.csv header")
                try: rows=[_parse_row(r) for r in reader]
                except (TypeError,ValueError,json.JSONDecodeError) as error: raise ValueError("invalid folds.csv row") from error
        _validate_rows(rows,manifest)
        return cls(root,manifest,rows)
    def save_rows(self,additions):
        candidate=[*self.rows,*additions]; _validate_rows(candidate,self.manifest); self.rows=candidate
        self.rows.sort(key=lambda r:(self.manifest["values"].index(r.parameter_value),r.fold_index,MODEL_NAMES.index(r.model))); lines=[";".join(_FOLDS_HEADER)]
        for r in self.rows: lines.append(";".join(json.dumps(list(getattr(r,name))) if name in _SAMPLE_FIELDS else str(getattr(r,name)) for name in _FOLDS_HEADER))
        _atomic(self.root/"folds.csv","\n".join(lines)+"\n")
    def save_failure(self,series,*,value,fold_index,seed,stage,error): _atomic(self.root/"failure.json",json.dumps({"series":series.name,"dataset":series.dataset,"parameter":series.parameter,"parameter_value":value,"fold_index":fold_index,"derived_seed":seed,"stage":stage,"exception_type":type(error).__name__,"exception_message":str(error)},sort_keys=True)+"\n")
    def clear_failure(self):
        try: (self.root/"failure.json").unlink()
        except FileNotFoundError: pass
def _mean_std(values:Sequence[float])->tuple[float,float]:
    mean=math.fsum(values)/len(values); return mean,(math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else 0.)
def aggregate_series_rows(rows:Iterable[SeriesFoldResult],*,values:Sequence[float],folds:int)->list[dict[str,float]]:
    all_rows=list(rows); result=[]
    for value in values:
        item={}
        for model in MODEL_NAMES:
            selected=[r for r in all_rows if r.parameter_value==value and r.model==model]
            if len(selected)!=folds or {r.fold_index for r in selected}!=set(range(folds)): raise ValueError("series is incomplete")
            for label,attr,divisor in _AGGREGATES:
                mean,std=_mean_std([getattr(r,attr)/divisor for r in selected]); item[f"{model}_{label}_mean"]=mean; item[f"{model}_{label}_std"]=std
        result.append(item)
    return result
def write_series_csv(series:ExperimentSeries,rows:Iterable[SeriesFoldResult],*,folds:int)->Path:
    header=(series.parameter,*SERIES_CSV_HEADER[1:]); aggregates=aggregate_series_rows(rows,values=series.values,folds=folds); lines=[";".join(header)]
    for value,item in zip(series.values,aggregates): lines.append(";".join([repr(value),*(repr(item[c]) for c in header[1:])]))
    target=series.output_dir/"series.csv"; _atomic(target,"\n".join(lines)+"\n"); return target
def run_series(series:ExperimentSeries,*,resume:bool=True,folds:int=SERIES_FOLDS)->Path|None:
    manifest=build_series_manifest(series,folds=folds,sha=implementation_fingerprint()); store=_SeriesStore.open(series,manifest=manifest,resume=resume); seeds=tuple(manifest["derived_generator_seeds"]); existing={(r.parameter_value,r.fold_index,r.model) for r in store.rows}
    for index,seed in enumerate(seeds):
        shared=None
        if series.parameter=="tau":
            try: shared=prepare_series_fold(series.dataset,seed=seed,train_censor_rate=.3)
            except BaseException as error: store.save_failure(series,value=None,fold_index=index,seed=seed,stage="prepare",error=error); raise
        for value in series.values:
            keys={(value,index,m) for m in MODEL_NAMES}
            if keys<=existing: continue
            if keys&existing: raise ValueError("incomplete persisted fold")
            started=time.monotonic(); stage="prepare"
            try:
                data=shared if shared is not None else prepare_series_fold(series.dataset,seed=seed,train_censor_rate=value); tau=value if series.parameter=="tau" else STARTING_STATES[series.dataset].tau; stage="evaluate"
                rows=tuple(replace(r,parameter_value=value) for r in evaluate_prepared_fold(data,tau=tau,fold_index=index,seed=seed))
                if len(rows)!=3 or {r.model for r in rows}!=set(MODEL_NAMES): raise ValueError("evaluator must return all three models")
                store.save_rows(rows); existing.update(keys); print(f"series={series.name} dataset={series.dataset} {series.parameter}={value} fold={index+1}/{folds} completed={len(existing)//3}/{len(series.values)*folds} elapsed={time.monotonic()-started:.1f}s",flush=True)
            except BaseException as error: store.save_failure(series,value=value,fold_index=index,seed=seed,stage=stage,error=error); print(f"series={series.name} {series.parameter}={value} fold={index+1}/{folds} failed: {error}",flush=True); raise
    result=write_series_csv(series,store.rows,folds=folds); store.clear_failure(); return result
def run_series_batch(series:Iterable[ExperimentSeries],*,resume:bool=True,folds:int=SERIES_FOLDS)->list[Path|None]:
    items=tuple(series); dirs=[item.output_dir.resolve() for item in items]
    if len(dirs)!=len(set(dirs)): raise ValueError("duplicate output directory in batch")
    return [run_series(s,resume=resume,folds=folds) for s in items]
