"""
evaluate.py  (final — multi-model with cross-model comparison)
==============================================================
Run for each model separately:
    python3 evaluate.py --experiments_dir ~/llm_sim/experiments
    python3 evaluate.py --experiments_dir ~/llm_sim/experiments_gemma
    python3 evaluate.py --experiments_dir ~/llm_sim/experiments_qwen

Then run comparison across all three:
    python3 evaluate.py --compare_all

Outputs per model (in experiments_*/plots/):
    predicted_vs_actual_*.png       scatter plots per experiment
    r2_by_model.png                 best R² per simulation model
    r2_by_dataset_size.png          dataset size effect
    r2_by_lora_rank.png             LoRA rank effect
    r2_by_learning_rate.png         learning rate effect
    r2_by_quantization.png          4-bit vs 8-bit
    r2_by_target_modules.png        attention vs MLP vs both
    r2_heatmap_lr_vs_rank.png       heatmap
    learning_curves_top.png         top 9 learning curves
    summary_table.csv               full results

Outputs for comparison (in ~/llm_sim/comparison/):
    comparison_r2_by_model.png      all 3 LLMs side by side per simulation
    comparison_r2_by_dataset_size.png
    comparison_best_configs.png     best config per LLM
    comparison_predicted_vs_actual_*.png  best experiment per LLM per sim model
    comparison_summary.csv          combined results all models
"""

import argparse, json, os, re, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

_USER = os.environ.get("USER", "user")
BASE_DIR = Path(f"/home/{_USER}/llm_sim")

SIM_MODELS = ["bank_renege","carwash","machine_shop","gas_station","movie_renege"]
SIM_LABELS = {
    "bank_renege":  "Bank Renege",
    "carwash":      "Carwash",
    "machine_shop": "Machine Shop",
    "gas_station":  "Gas Station",
    "movie_renege": "Movie Renege",
}
MODEL_COLORS = {
    "bank_renege":  "#378ADD",
    "carwash":      "#1D9E75",
    "machine_shop": "#D85A30",
    "gas_station":  "#BA7517",
    "movie_renege": "#7F77DD",
}
LLM_COLORS = {
    "llama":  "#378ADD",
    "gemma":  "#1D9E75",
    "qwen":   "#D85A30",
}
LLM_LABELS = {
    "llama":  "LLaMA 3.2-1B",
    "gemma":  "Gemma 3-1B",
    "qwen":   "Qwen 2.5-1.5B",
}
DATASET_DIR = Path(f"/home/{_USER}/DES-FineTuning")
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
STYLE = dict(dpi=150, bbox_inches="tight")

plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_data():
    with open(DATASET_DIR / "dataset_test.json") as f:
        records = json.load(f)
    return [r for r in records if r["model"] in SIM_MODELS]

def format_prompt(question):
    return f"### Simulation Question:\n{question}\n\n### Answer:\n"

def extract_number(text):
    matches = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
    if matches:
        try: return float(matches[0])
        except: pass
    return None

def load_log(exp_dir):
    p = exp_dir / "training_log.json"
    if not p.exists(): return None
    with open(p) as f: return json.load(f)

def get_final_val_loss(log):
    return log["eval_losses"][-1]["eval_loss"] if log.get("eval_losses") else float("inf")

def get_all_completed(experiments_dir):
    results = []
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir() or not (d/"DONE").exists(): continue
        log = load_log(d)
        if log: results.append((d.name, d, log))
    return results

def detect_llm_family(experiments_dir):
    """Detect LLM family from experiment logs."""
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir(): continue
        log = load_log(d)
        if log:
            return log.get("model_family", "unknown")
    return "unknown"


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(exp_dir, test_records):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = exp_dir / "adapter"
    if not adapter_path.exists(): return []
    log = load_log(exp_dir)
    base_model_id = log.get("base_model_id", "meta-llama/Llama-3.2-1B")
    quant = log["config"]["quantization"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if quant == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    results = []
    prompts = [format_prompt(r["question"]) for r in test_records]
    for i in range(0, len(prompts), BATCH_SIZE):
        bp = prompts[i:i+BATCH_SIZE]
        bt = [r["answer"] for r in test_records[i:i+BATCH_SIZE]]
        bs = [r["model"]  for r in test_records[i:i+BATCH_SIZE]]
        enc = tokenizer(bp, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        for oids, tv, sm in zip(out, bt, bs):
            text = tokenizer.decode(oids[enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            results.append({"model":sm,"true":tv,"pred":extract_number(text),"text":text})
        if (i//BATCH_SIZE)%10==0:
            print(f"    {min(i+BATCH_SIZE,len(prompts))}/{len(prompts)}")

    del model
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()
    return results


def compute_metrics(results):
    valid = [r for r in results if r["pred"] is not None]
    if len(valid)<2:
        return {"r2":None,"mae":None,"rmse":None,
                "n_valid":len(valid),"n_total":len(results),"by_model":{}}
    t = np.array([r["true"] for r in valid])
    p = np.array([r["pred"] for r in valid])
    by_model = {}
    for sim in SIM_MODELS:
        sub = [r for r in valid if r["model"]==sim]
        if len(sub)<2: by_model[sim]={"r2":None,"mae":None,"rmse":None}; continue
        ts=np.array([r["true"] for r in sub]); ps=np.array([r["pred"] for r in sub])
        by_model[sim]={"r2":float(r2_score(ts,ps)),
                       "mae":float(mean_absolute_error(ts,ps)),
                       "rmse":float(np.sqrt(mean_squared_error(ts,ps))),"n":len(sub)}
    return {"r2":float(r2_score(t,p)),"mae":float(mean_absolute_error(t,p)),
            "rmse":float(np.sqrt(mean_squared_error(t,p))),
            "n_valid":len(valid),"n_total":len(results),"by_model":by_model}


# ── Per-model plots ────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(results, exp_id, metrics, plots_dir):
    fig, axes = plt.subplots(1,5,figsize=(20,4))
    fig.suptitle(f"Predicted vs Actual — {exp_id}", fontsize=9, y=1.02)
    for ax, sim in zip(axes, SIM_MODELS):
        sub = [r for r in results if r["model"]==sim and r["pred"] is not None]
        if not sub: ax.set_title(SIM_LABELS[sim]); continue
        t=np.array([r["true"] for r in sub]); p=np.array([r["pred"] for r in sub])
        r2=metrics["by_model"].get(sim,{}).get("r2")
        ax.scatter(t,p,alpha=0.4,s=15,color=MODEL_COLORS[sim],edgecolors="none")
        mn,mx=min(t.min(),p.min()),max(t.max(),p.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
        ax.set_xlabel("Actual",fontsize=8); ax.set_ylabel("Predicted",fontsize=8)
        r2_str=f"R²={r2:.3f}" if r2 is not None else "R²=N/A"
        ax.set_title(f"{SIM_LABELS[sim]}\n{r2_str}",fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True,alpha=0.2)
    plt.tight_layout()
    path = plots_dir / f"predicted_vs_actual_{exp_id[:40]}.png"
    fig.savefig(path,**STYLE); plt.close(fig)

def plot_bar(df, col, title, xlabel, filename, plots_dir, order=None):
    grp=df.groupby(col)["r2"].agg(["mean","std"]).reset_index()
    if order: grp=grp.set_index(col).reindex(order).reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(grp[col].astype(str),grp["mean"],yerr=grp["std"],capsize=5,
           color="#378ADD",edgecolor="white",linewidth=0.5,error_kw=dict(lw=1))
    ax.set_ylim(0,min(1.1,(grp["mean"]+grp["std"]).max()+0.15))
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean R²")
    ax.set_title(title); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); fig.savefig(plots_dir/filename,**STYLE); plt.close(fig)

def plot_r2_by_model(df, plots_dir):
    best=df.groupby("sim_model")["r2"].max().reindex(SIM_MODELS)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar([SIM_LABELS[m] for m in SIM_MODELS],best.values,
                color=[MODEL_COLORS[m] for m in SIM_MODELS],edgecolor="white")
    for bar,val in zip(bars,best.values):
        if val is not None and not np.isnan(val):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                    f"{val:.3f}",ha="center",va="bottom",fontsize=9)
    ax.set_ylim(0,1.1); ax.set_ylabel("Best R²")
    ax.set_title("Best R² by Simulation Model"); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); fig.savefig(plots_dir/"r2_by_model.png",**STYLE); plt.close(fig)

def plot_quantization(df, plots_dir):
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(SIM_MODELS)); w=0.35
    for i,(quant,color) in enumerate([("4bit","#378ADD"),("8bit","#D85A30")]):
        means=[df[(df["sim_model"]==s)&(df["quantization"]==quant)]["r2"].mean() for s in SIM_MODELS]
        ax.bar(x+i*w-w/2,means,w,label=quant,color=color,edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([SIM_LABELS[m] for m in SIM_MODELS],fontsize=9)
    ax.set_ylim(0,1.1); ax.set_ylabel("Mean R²")
    ax.set_title("4-bit vs 8-bit Quantization"); ax.legend(); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); fig.savefig(plots_dir/"r2_by_quantization.png",**STYLE); plt.close(fig)

def plot_target_modules(df, plots_dir):
    ranks=sorted(df["lora_r"].unique())
    colors={"attention":"#378ADD","mlp":"#1D9E75","both":"#D85A30"}
    fig,ax=plt.subplots(figsize=(8,5))
    for tm,color in colors.items():
        means=[df[(df["target_modules"]==tm)&(df["lora_r"]==r)]["r2"].mean() for r in ranks]
        ax.plot([f"r={r}" for r in ranks],means,marker="o",label=tm,color=color,lw=2,ms=7)
    ax.set_ylim(0,1); ax.set_xlabel("LoRA Rank"); ax.set_ylabel("Mean R²")
    ax.set_title("R² vs LoRA Rank by Target Modules"); ax.legend(); ax.grid(True,alpha=0.25)
    plt.tight_layout(); fig.savefig(plots_dir/"r2_by_target_modules.png",**STYLE); plt.close(fig)

def plot_heatmap(df, plots_dir):
    lrs=sorted(df["learning_rate"].unique()); ranks=sorted(df["lora_r"].unique())
    matrix=np.full((len(lrs),len(ranks)),np.nan)
    for i,lr in enumerate(lrs):
        for j,r in enumerate(ranks):
            sub=df[(df["learning_rate"]==lr)&(df["lora_r"]==r)]["r2"]
            if len(sub)>0: matrix[i,j]=sub.mean()
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(matrix,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(range(len(ranks))); ax.set_xticklabels([f"r={r}" for r in ranks])
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"lr={lr:.0e}" for lr in lrs])
    ax.set_title("Mean R²: Learning Rate × LoRA Rank")
    plt.colorbar(im,ax=ax,label="Mean R²")
    for i in range(len(lrs)):
        for j in range(len(ranks)):
            v=matrix[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=10,
                        color="white" if v<0.5 else "black")
    plt.tight_layout(); fig.savefig(plots_dir/"r2_heatmap_lr_vs_rank.png",**STYLE); plt.close(fig)

def plot_learning_curves(all_exps, plots_dir, top_n=9):
    ranked=sorted(all_exps,key=lambda x: get_final_val_loss(x[2]))[:top_n]
    ncols=3; nrows=(len(ranked)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(14,nrows*3.5))
    axes=axes.flatten()
    fig.suptitle(f"Learning Curves — Top {top_n} by Val Loss",fontsize=11)
    for ax,(exp_id,exp_dir,log) in zip(axes,ranked):
        ts=[x["step"] for x in log["train_losses"]]; tl=[x["loss"] for x in log["train_losses"]]
        es=[x["step"] for x in log["eval_losses"]];  el=[x["eval_loss"] for x in log["eval_losses"]]
        ax.plot(ts,tl,color="#378ADD",lw=1.5,label="train")
        ax.plot(es,el,color="#D85A30",lw=2,ls="--",marker="o",ms=3,label="val")
        ax.set_title(exp_id[:35],fontsize=7); ax.tick_params(labelsize=7)
        ax.grid(True,alpha=0.2); ax.legend(fontsize=6)
    for ax in axes[len(ranked):]: ax.set_visible(False)
    plt.tight_layout(); fig.savefig(plots_dir/"learning_curves_top.png",**STYLE); plt.close(fig)


# ── Cross-model comparison plots ──────────────────────────────────────────────

def plot_comparison_r2_by_sim_model(dfs_by_family, comp_dir):
    """Grouped bar: best R² per simulation model, grouped by LLM."""
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(SIM_MODELS))
    families=list(dfs_by_family.keys())
    w=0.25
    for i,family in enumerate(families):
        df=dfs_by_family[family]
        best=[df[df["sim_model"]==sim]["r2"].max() for sim in SIM_MODELS]
        offset=(i-(len(families)-1)/2)*w
        bars=ax.bar(x+offset,best,w,label=LLM_LABELS.get(family,family),
                    color=LLM_COLORS.get(family,"#888780"),edgecolor="white",linewidth=0.5)
        for bar,val in zip(bars,best):
            if val is not None and not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                        f"{val:.2f}",ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([SIM_LABELS[m] for m in SIM_MODELS],fontsize=10)
    ax.set_ylim(0,1.15); ax.set_ylabel("Best R²")
    ax.set_title("Best R² by Simulation Model — LLM Comparison")
    ax.legend(fontsize=10); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); fig.savefig(comp_dir/"comparison_r2_by_sim_model.png",**STYLE); plt.close(fig)

def plot_comparison_r2_by_dataset_size(dfs_by_family, comp_dir):
    """Line plot: mean R² vs dataset size for each LLM."""
    fig,ax=plt.subplots(figsize=(9,5))
    for family,df in dfs_by_family.items():
        sizes=sorted(df["dataset_size"].unique())
        means=[df[df["dataset_size"]==ds]["r2"].mean() for ds in sizes]
        ax.plot(sizes,means,marker="o",label=LLM_LABELS.get(family,family),
                color=LLM_COLORS.get(family,"#888780"),lw=2.5,ms=8)
    ax.set_xlabel("Training examples per simulation model")
    ax.set_ylabel("Mean R²"); ax.set_title("Effect of Dataset Size — LLM Comparison")
    ax.legend(fontsize=10); ax.grid(True,alpha=0.25)
    plt.tight_layout(); fig.savefig(comp_dir/"comparison_r2_by_dataset_size.png",**STYLE); plt.close(fig)

def plot_comparison_best_configs(dfs_by_family, comp_dir):
    """Table-style bar showing best overall config per LLM."""
    fig,axes=plt.subplots(1,len(dfs_by_family),figsize=(14,5),sharey=True)
    fig.suptitle("Best Configuration per LLM — R² by Simulation Model",fontsize=12)
    for ax,(family,df) in zip(axes,dfs_by_family.items()):
        # Find best experiment (highest mean R² across sim models)
        best_exp=df.groupby("exp_id")["r2"].mean().idxmax()
        best_df=df[df["exp_id"]==best_exp]
        r2_vals=[best_df[best_df["sim_model"]==sim]["r2"].values[0]
                 if len(best_df[best_df["sim_model"]==sim])>0 else 0
                 for sim in SIM_MODELS]
        bars=ax.bar([SIM_LABELS[m].replace(" ","\n") for m in SIM_MODELS],
                    r2_vals,color=[MODEL_COLORS[m] for m in SIM_MODELS],edgecolor="white")
        for bar,val in zip(bars,r2_vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                    f"{val:.3f}",ha="center",va="bottom",fontsize=8)
        ax.set_ylim(0,1.1); ax.set_title(f"{LLM_LABELS.get(family,family)}\n{best_exp[:30]}",fontsize=8)
        ax.set_ylabel("R²"); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); fig.savefig(comp_dir/"comparison_best_configs.png",**STYLE); plt.close(fig)

def plot_comparison_predicted_vs_actual_best(dfs_by_family, results_by_family, comp_dir):
    """For each simulation model, show predicted vs actual for best experiment per LLM."""
    for sim in SIM_MODELS:
        fig,axes=plt.subplots(1,len(dfs_by_family),figsize=(15,4))
        fig.suptitle(f"Predicted vs Actual — {SIM_LABELS[sim]} — Best Config per LLM",
                     fontsize=10,y=1.02)
        for ax,(family,df) in zip(axes,dfs_by_family.items()):
            # Find best experiment for this sim model
            sim_df=df[df["sim_model"]==sim]
            if len(sim_df)==0 or sim_df["r2"].isna().all():
                ax.set_title(f"{LLM_LABELS.get(family,family)}\nNo data"); continue
            best_exp=sim_df.loc[sim_df["r2"].idxmax(),"exp_id"]
            # Get inference results
            results=results_by_family.get(family,{}).get(best_exp,[])
            sim_results=[r for r in results if r["model"]==sim and r["pred"] is not None]
            if not sim_results:
                ax.set_title(f"{LLM_LABELS.get(family,family)}\nNo predictions"); continue
            t=np.array([r["true"] for r in sim_results])
            p=np.array([r["pred"] for r in sim_results])
            r2=float(r2_score(t,p)) if len(t)>=2 else None
            ax.scatter(t,p,alpha=0.4,s=15,color=LLM_COLORS.get(family,"#888780"),edgecolors="none")
            mn,mx=min(t.min(),p.min()),max(t.max(),p.max())
            ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
            ax.set_xlabel("Actual",fontsize=8); ax.set_ylabel("Predicted",fontsize=8)
            r2_str=f"R²={r2:.3f}" if r2 is not None else "N/A"
            ax.set_title(f"{LLM_LABELS.get(family,family)}\n{r2_str}",fontsize=9)
            ax.tick_params(labelsize=7); ax.grid(True,alpha=0.2)
        plt.tight_layout()
        fig.savefig(comp_dir/f"comparison_predicted_vs_actual_{sim}.png",**STYLE); plt.close(fig)

def plot_comparison_lora_rank(dfs_by_family, comp_dir):
    """Line plot: mean R² vs LoRA rank for each LLM."""
    fig,ax=plt.subplots(figsize=(9,5))
    ranks=[4,8,16,32]
    for family,df in dfs_by_family.items():
        means=[df[df["lora_r"]==r]["r2"].mean() for r in ranks]
        ax.plot([f"r={r}" for r in ranks],means,marker="o",
                label=LLM_LABELS.get(family,family),
                color=LLM_COLORS.get(family,"#888780"),lw=2.5,ms=8)
    ax.set_ylim(0,1); ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Mean R²"); ax.set_title("LoRA Rank Effect — LLM Comparison")
    ax.legend(fontsize=10); ax.grid(True,alpha=0.25)
    plt.tight_layout(); fig.savefig(comp_dir/"comparison_lora_rank.png",**STYLE); plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate_one(experiments_dir, args):
    """Run evaluation for one model's experiment directory."""
    experiments_dir = Path(experiments_dir)
    plots_dir = experiments_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    llm_family = detect_llm_family(experiments_dir)
    print(f"\n{'='*60}")
    print(f"  Evaluating: {experiments_dir.name}")
    print(f"  LLM family: {llm_family}")
    print(f"  Plots dir:  {plots_dir}")
    print(f"{'='*60}\n")

    all_exps = get_all_completed(experiments_dir)
    print(f"Completed experiments: {len(all_exps)}")
    if args.exp_id:   all_exps=[e for e in all_exps if e[0]==args.exp_id]
    elif args.top_n:  all_exps=sorted(all_exps,key=lambda x: get_final_val_loss(x[2]))[:args.top_n]

    test_records = load_test_data()
    print(f"Test records: {len(test_records)}")

    cache_path = experiments_dir / "all_inference_results.json"
    all_rows = []
    cached_results = {}  # exp_id -> results list

    if args.skip_inference and cache_path.exists():
        print("Loading cached results...")
        with open(cache_path) as f:
            cached = json.load(f)
        all_rows = cached["rows"]
        cached_results = cached.get("results_by_exp", {})
    else:
        for i,(exp_id,exp_dir,log) in enumerate(all_exps):
            print(f"\n[{i+1}/{len(all_exps)}] {exp_id}")
            cfg = log["config"]
            inf_path = exp_dir / "inference_results.json"

            if inf_path.exists():
                with open(inf_path) as f: saved=json.load(f)
                metrics=saved["metrics"]; results=saved["results"]
                print(f"  [CACHED] R²={metrics.get('r2')}")
            else:
                results = run_inference(exp_dir, test_records)
                metrics = compute_metrics(results)
                with open(inf_path,"w") as f:
                    json.dump({"metrics":metrics,"results":results},f)
                if results:
                    plot_predicted_vs_actual(results,exp_id,metrics,plots_dir)
                print(f"  R²={metrics.get('r2')} | {metrics.get('n_valid')}/{metrics.get('n_total')} valid")

            cached_results[exp_id] = results

            for sim in SIM_MODELS:
                m=metrics["by_model"].get(sim,{})
                all_rows.append({
                    "exp_id":         exp_id,
                    "sim_model":      sim,
                    "base_model_id":  log.get("base_model_id","unknown"),
                    "model_family":   log.get("model_family", llm_family),
                    "dataset_size":   cfg["dataset_size"],
                    "quantization":   cfg["quantization"],
                    "lora_r":         cfg["lora_r"],
                    "lora_alpha":     cfg["lora_alpha"],
                    "lora_dropout":   cfg["lora_dropout"],
                    "target_modules": cfg["target_modules"],
                    "learning_rate":  cfg["learning_rate"],
                    "num_epochs":     cfg["num_epochs"],
                    "r2":             m.get("r2"),
                    "mae":            m.get("mae"),
                    "rmse":           m.get("rmse"),
                    "val_loss":       get_final_val_loss(log),
                    "train_runtime_s":log.get("train_runtime_s"),
                })

        with open(cache_path,"w") as f:
            json.dump({"rows":all_rows,"results_by_exp":cached_results},f)

    df = pd.DataFrame(all_rows).dropna(subset=["r2"])
    df.to_csv(plots_dir/"summary_table.csv",index=False)
    print(f"\nGenerating plots...")

    plot_r2_by_model(df,plots_dir);                          print("  v r2_by_model.png")
    plot_bar(df,"dataset_size","Dataset Size Effect","Samples per model",
             "r2_by_dataset_size.png",plots_dir,
             order=sorted(df["dataset_size"].unique()));      print("  v r2_by_dataset_size.png")
    plot_bar(df,"lora_r","LoRA Rank Effect","LoRA rank r",
             "r2_by_lora_rank.png",plots_dir,order=[4,8,16,32]); print("  v r2_by_lora_rank.png")
    plot_bar(df,"learning_rate","Learning Rate Effect","Learning rate",
             "r2_by_learning_rate.png",plots_dir);            print("  v r2_by_learning_rate.png")
    plot_quantization(df,plots_dir);                          print("  v r2_by_quantization.png")
    plot_target_modules(df,plots_dir);                        print("  v r2_by_target_modules.png")
    plot_heatmap(df,plots_dir);                               print("  v r2_heatmap_lr_vs_rank.png")
    plot_learning_curves(all_exps,plots_dir);                 print("  v learning_curves_top.png")

    print(f"\nTop 10 by mean R²:")
    top10=df.groupby("exp_id")["r2"].mean().sort_values(ascending=False).head(10)
    for eid,r2 in top10.items(): print(f"  {r2:.4f}  {eid}")
    print(f"\nAll plots saved to: {plots_dir}")
    return df, cached_results, llm_family


def compare_all(args):
    """Load results from all three models and generate comparison plots."""
    exp_dirs = {
        "llama": BASE_DIR / "experiments",
        "gemma": BASE_DIR / "experiments_gemma",
        "qwen":  BASE_DIR / "experiments_qwen",
    }
    comp_dir = BASE_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    dfs_by_family = {}
    results_by_family = {}

    for family, exp_dir in exp_dirs.items():
        cache_path = exp_dir / "all_inference_results.json"
        if not exp_dir.exists():
            print(f"  [SKIP] {family} — directory not found: {exp_dir}")
            continue
        if not cache_path.exists():
            print(f"  [SKIP] {family} — no cached results yet. Run evaluate.py --experiments_dir {exp_dir} first.")
            continue
        print(f"  Loading {family} results...")
        with open(cache_path) as f: cached=json.load(f)
        df = pd.DataFrame(cached["rows"]).dropna(subset=["r2"])
        df["model_family"] = family
        dfs_by_family[family] = df
        results_by_family[family] = cached.get("results_by_exp", {})
        print(f"    {len(df)} rows, {df['exp_id'].nunique()} experiments")

    if len(dfs_by_family) < 2:
        print("Need at least 2 models evaluated to compare. Run evaluate.py for each model first.")
        return

    # Combined CSV
    combined = pd.concat(dfs_by_family.values(), ignore_index=True)
    combined.to_csv(comp_dir/"comparison_summary.csv", index=False)
    print(f"\nCombined CSV: {comp_dir}/comparison_summary.csv")
    print(f"  Total rows: {len(combined)}")

    print("\nGenerating comparison plots...")
    plot_comparison_r2_by_sim_model(dfs_by_family, comp_dir)
    print("  v comparison_r2_by_sim_model.png")
    plot_comparison_r2_by_dataset_size(dfs_by_family, comp_dir)
    print("  v comparison_r2_by_dataset_size.png")
    plot_comparison_best_configs(dfs_by_family, comp_dir)
    print("  v comparison_best_configs.png")
    plot_comparison_predicted_vs_actual_best(dfs_by_family, results_by_family, comp_dir)
    print("  v comparison_predicted_vs_actual_*.png (one per simulation model)")
    plot_comparison_lora_rank(dfs_by_family, comp_dir)
    print("  v comparison_lora_rank.png")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  Cross-model summary — best R² per simulation model")
    print(f"{'='*65}")
    print(f"  {'Simulation':<20} {'LLaMA':>10} {'Gemma':>10} {'Qwen':>10}")
    print(f"  {'-'*50}")
    for sim in SIM_MODELS:
        row = f"  {SIM_LABELS[sim]:<20}"
        for family in ["llama","gemma","qwen"]:
            if family in dfs_by_family:
                val=dfs_by_family[family][dfs_by_family[family]["sim_model"]==sim]["r2"].max()
                row += f" {val:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print(f"\nAll comparison plots saved to: {comp_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--compare_all",     action="store_true",
                        help="Generate cross-model comparison plots")
    parser.add_argument("--exp_id",          type=str, default=None)
    parser.add_argument("--top_n",           type=int, default=None)
    parser.add_argument("--skip_inference",  action="store_true")
    args = parser.parse_args()

    if args.compare_all:
        compare_all(args)
        return

    if args.experiments_dir:
        evaluate_one(args.experiments_dir, args)
    else:
        # Default: evaluate LLaMA
        evaluate_one(str(BASE_DIR / "experiments"), args)


if __name__ == "__main__":
    main()
