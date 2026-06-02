"""
evaluate.py  (v2 — multi-model support)
========================================
Reads base_model_id from each experiment's training_log.json
so it works correctly for LLaMA, Gemma, and Qwen experiments.

Usage:
    python3 evaluate.py
    python3 evaluate.py --experiments_dir ~/llm_sim/experiments_gemma
    python3 evaluate.py --experiments_dir ~/llm_sim/experiments_qwen
    python3 evaluate.py --skip_inference   # regenerate plots only
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

SIM_MODELS = ["bank_renege","carwash","machine_shop","gas_station","movie_renege"]
MODEL_COLORS = {
    "bank_renege":  "#378ADD",
    "carwash":      "#1D9E75",
    "machine_shop": "#D85A30",
    "gas_station":  "#BA7517",
    "movie_renege": "#7F77DD",
}
DATASET_DIR = Path(f"/home/{_USER}/DES-FineTuning")
MAX_SEQ_LENGTH = 256
BATCH_SIZE     = 8
STYLE = dict(dpi=150, bbox_inches="tight")

plt.rcParams.update({
    "font.family":     "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "grid.linestyle":  "--",
})


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


def run_inference(exp_dir, test_records):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = exp_dir / "adapter"
    if not adapter_path.exists(): return []

    log = load_log(exp_dir)

    # Read base model from training log (supports LLaMA, Gemma, Qwen)
    base_model_id = log.get("base_model_id", f"meta-llama/Llama-3.2-1B")
    quant         = log["config"]["quantization"]

    print(f"    Base model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb,
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
            text = tokenizer.decode(
                oids[enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
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


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(results, exp_id, metrics, plots_dir):
    fig, axes = plt.subplots(1, len(SIM_MODELS), figsize=(18,4))
    fig.suptitle(f"Predicted vs Actual — {exp_id}", fontsize=9, y=1.02)
    for ax, sim in zip(axes, SIM_MODELS):
        sub = [r for r in results if r["model"]==sim and r["pred"] is not None]
        if not sub: ax.set_title(sim.replace("_"," ")); continue
        t=np.array([r["true"] for r in sub]); p=np.array([r["pred"] for r in sub])
        r2=metrics["by_model"].get(sim,{}).get("r2")
        ax.scatter(t,p,alpha=0.4,s=15,color=MODEL_COLORS[sim],edgecolors="none")
        mn,mx=min(t.min(),p.min()),max(t.max(),p.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.4)
        ax.set_xlabel("Actual",fontsize=8); ax.set_ylabel("Predicted",fontsize=8)
        ax.set_title(f"{sim.replace('_',' ')}\nR²={r2:.3f}" if r2 else sim.replace("_"," "),fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True,alpha=0.2)
    plt.tight_layout()
    path = plots_dir / f"predicted_vs_actual_{exp_id[:40]}.png"
    fig.savefig(path,**STYLE); plt.close(fig); return path

def plot_bar(df, col, title, xlabel, filename, plots_dir, order=None):
    grp=df.groupby(col)["r2"].agg(["mean","std"]).reset_index()
    if order: grp=grp.set_index(col).reindex(order).reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(grp[col].astype(str),grp["mean"],yerr=grp["std"],capsize=5,
           color="#378ADD",edgecolor="white",linewidth=0.5,error_kw=dict(lw=1))
    ax.set_ylim(0,min(1.1,(grp["mean"]+grp["std"]).max()+0.15))
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean R²"); ax.set_title(title); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); path=plots_dir/filename; fig.savefig(path,**STYLE); plt.close(fig); return path

def plot_r2_by_model(df, plots_dir):
    best=df.groupby("sim_model")["r2"].max().reindex(SIM_MODELS)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar([m.replace("_","\n") for m in SIM_MODELS],best.values,
                color=[MODEL_COLORS[m] for m in SIM_MODELS],edgecolor="white")
    for bar,val in zip(bars,best.values):
        if not np.isnan(val):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                    f"{val:.3f}",ha="center",va="bottom",fontsize=9)
    ax.set_ylim(0,1.1); ax.set_ylabel("Best R²"); ax.set_title("Best R² by Simulation Model"); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); path=plots_dir/"r2_by_model.png"; fig.savefig(path,**STYLE); plt.close(fig); return path

def plot_quantization(df, plots_dir):
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(SIM_MODELS)); w=0.35
    for i,(quant,color) in enumerate([("4bit","#378ADD"),("8bit","#D85A30")]):
        means=[df[(df["sim_model"]==s)&(df["quantization"]==quant)]["r2"].mean() for s in SIM_MODELS]
        ax.bar(x+i*w-w/2,means,w,label=quant,color=color,edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_","\n") for m in SIM_MODELS],fontsize=9)
    ax.set_ylim(0,1.1); ax.set_ylabel("Mean R²"); ax.set_title("4-bit vs 8-bit by Simulation Model")
    ax.legend(); ax.grid(True,axis="y",alpha=0.25)
    plt.tight_layout(); path=plots_dir/"r2_by_quantization.png"; fig.savefig(path,**STYLE); plt.close(fig); return path

def plot_target_modules(df, plots_dir):
    ranks=sorted(df["lora_r"].unique())
    colors={"attention":"#378ADD","mlp":"#1D9E75","both":"#D85A30"}
    fig,ax=plt.subplots(figsize=(8,5))
    for tm,color in colors.items():
        means=[df[(df["target_modules"]==tm)&(df["lora_r"]==r)]["r2"].mean() for r in ranks]
        ax.plot([f"r={r}" for r in ranks],means,marker="o",label=tm,color=color,lw=2,ms=7)
    ax.set_ylim(0,1); ax.set_xlabel("LoRA Rank"); ax.set_ylabel("Mean R²")
    ax.set_title("R² vs LoRA Rank by Target Modules"); ax.legend(); ax.grid(True,alpha=0.25)
    plt.tight_layout(); path=plots_dir/"r2_by_target_modules.png"; fig.savefig(path,**STYLE); plt.close(fig); return path

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
    ax.set_title("Mean R²: Learning Rate × LoRA Rank"); plt.colorbar(im,ax=ax,label="Mean R²")
    for i in range(len(lrs)):
        for j in range(len(ranks)):
            v=matrix[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=10,
                        color="white" if v<0.5 else "black")
    plt.tight_layout(); path=plots_dir/"r2_heatmap_lr_vs_rank.png"; fig.savefig(path,**STYLE); plt.close(fig); return path

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
    plt.tight_layout(); path=plots_dir/"learning_curves_top.png"; fig.savefig(path,**STYLE); plt.close(fig); return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", type=str, default=None,
                        help="Path to experiments directory (default: ~/llm_sim/experiments)")
    parser.add_argument("--exp_id",          type=str, default=None)
    parser.add_argument("--top_n",           type=int, default=None)
    parser.add_argument("--skip_inference",  action="store_true")
    args = parser.parse_args()

    if args.experiments_dir:
        experiments_dir = Path(args.experiments_dir)
    else:
        experiments_dir = Path(f"/home/{_USER}/llm_sim/experiments")

    plots_dir = experiments_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 3 — Evaluation")
    print(f"  Experiments dir : {experiments_dir}")
    print(f"  Plots dir       : {plots_dir}")
    print(f"{'='*60}\n")

    all_exps = get_all_completed(experiments_dir)
    print(f"Completed experiments: {len(all_exps)}")

    if args.exp_id:   all_exps=[e for e in all_exps if e[0]==args.exp_id]
    elif args.top_n:  all_exps=sorted(all_exps,key=lambda x: get_final_val_loss(x[2]))[:args.top_n]

    test_records = load_test_data()
    print(f"Test records: {len(test_records)}")

    cache_path = experiments_dir / "all_inference_results.json"
    all_rows   = []

    if args.skip_inference and cache_path.exists():
        print("Loading cached results...")
        with open(cache_path) as f: all_rows=json.load(f)
    else:
        for i,(exp_id,exp_dir,log) in enumerate(all_exps):
            print(f"\n[{i+1}/{len(all_exps)}] {exp_id}")
            cfg=log["config"]

            # Skip if already evaluated
            inf_path = exp_dir / "inference_results.json"
            if inf_path.exists():
                with open(inf_path) as f:
                    saved = json.load(f)
                metrics = saved["metrics"]
                results = saved["results"]
                print(f"  [CACHED] R²={metrics.get('r2')}")
            else:
                results = run_inference(exp_dir, test_records)
                metrics = compute_metrics(results)
                with open(inf_path,"w") as f:
                    json.dump({"metrics":metrics,"results":results},f)
                if results:
                    plot_predicted_vs_actual(results,exp_id,metrics,plots_dir)
                print(f"  R²={metrics.get('r2')} | {metrics.get('n_valid')}/{metrics.get('n_total')} valid")

            for sim in SIM_MODELS:
                m=metrics["by_model"].get(sim,{})
                all_rows.append({
                    "exp_id":          exp_id,
                    "sim_model":       sim,
                    "base_model_id":   log.get("base_model_id","unknown"),
                    "model_family":    log.get("model_family","unknown"),
                    "dataset_size":    cfg["dataset_size"],
                    "quantization":    cfg["quantization"],
                    "lora_r":          cfg["lora_r"],
                    "lora_alpha":      cfg["lora_alpha"],
                    "lora_dropout":    cfg["lora_dropout"],
                    "target_modules":  cfg["target_modules"],
                    "learning_rate":   cfg["learning_rate"],
                    "num_epochs":      cfg["num_epochs"],
                    "r2":              m.get("r2"),
                    "mae":             m.get("mae"),
                    "rmse":            m.get("rmse"),
                    "val_loss":        get_final_val_loss(log),
                    "train_runtime_s": log.get("train_runtime_s"),
                })

        with open(cache_path,"w") as f: json.dump(all_rows,f)

    df=pd.DataFrame(all_rows).dropna(subset=["r2"])
    df.to_csv(plots_dir/"summary_table.csv",index=False)
    print(f"\nGenerating summary plots...")

    plot_r2_by_model(df, plots_dir);              print("  ✓ r2_by_model.png")
    plot_bar(df,"dataset_size","Dataset Size Effect","Samples per model",
             "r2_by_dataset_size.png",plots_dir,
             order=sorted(df["dataset_size"].unique())); print("  ✓ r2_by_dataset_size.png")
    plot_bar(df,"lora_r","LoRA Rank Effect","LoRA rank r",
             "r2_by_lora_rank.png",plots_dir,order=[4,8,16,32]); print("  ✓ r2_by_lora_rank.png")
    plot_bar(df,"learning_rate","Learning Rate Effect","Learning rate",
             "r2_by_learning_rate.png",plots_dir);  print("  ✓ r2_by_learning_rate.png")
    plot_quantization(df, plots_dir);              print("  ✓ r2_by_quantization.png")
    plot_target_modules(df, plots_dir);            print("  ✓ r2_by_target_modules.png")
    plot_heatmap(df, plots_dir);                   print("  ✓ r2_heatmap_lr_vs_rank.png")
    plot_learning_curves(all_exps, plots_dir);     print("  ✓ learning_curves_top.png")

    print(f"\nTop 10 by mean R²:")
    top10=df.groupby("exp_id")["r2"].mean().sort_values(ascending=False).head(10)
    for eid,r2 in top10.items(): print(f"  {r2:.4f}  {eid}")
    print(f"\nAll plots → {plots_dir}\n")


if __name__=="__main__":
    main()
