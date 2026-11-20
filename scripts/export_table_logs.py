import csv, json, os, sys
from pathlib import Path

RUNS_DIR = Path('/content/drive/MyDrive/thesis/runs')
OUT_DIR  = Path('/content/logs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- classifier ---
def ckpt_to_init(ckpt):
    if not ckpt: return "random"
    s = ckpt.lower()
    if "ssl_casia" in s and "graph" in s: return "casia_graph"
    if "ssl_casia" in s and "spatial" in s and "graph" not in s: return "casia_spatial_only"
    if "ssl_casia" in s: return "casia"
    if "densegait" in s and ("sub" in s or "8140" in s): return "densegait_sub"
    if "densegait" in s: return "densegait"
    return "unknown"

def classify(cfg):
    tags = []
    init = ckpt_to_init(cfg.get("ckpt"))
    region = cfg.get("region", "R1")
    protocol = cfg.get("protocol", "")
    freeze = bool(cfg.get("freeze_backbone", 0))
    vm = bool(cfg.get("knn_view_match", 0))
    try:
        is_main_lr = abs(float(cfg.get("lr_backbone", 0)) - 1e-4) < 1e-9
    except Exception:
        is_main_lr = False
    std = (protocol == "standard")
    if std and region=="R1" and not freeze and not vm and init in ("casia","densegait","random") and is_main_lr: tags.append("table02")
    if std and region=="R1" and not freeze and not vm and init in ("casia","densegait") and is_main_lr: tags.append("table05")
    if std and region=="R1" and init=="casia" and cfg.get("seed",-1)==0 and not freeze: tags.append("table06")
    if std and region=="R1" and init=="casia_graph": tags.append("table08")
    if std and region=="R1" and init=="casia_spatial_only": tags.append("table09")
    if std and region in ("R1","R2") and not freeze and not vm and init in ("casia","random") and is_main_lr: tags.append("table10")
    if std and region=="R1" and not freeze and not vm and init in ("casia","densegait","densegait_sub") and is_main_lr: tags.append("table11")
    return tags, init

# --- read runs ---
if not RUNS_DIR.exists():
    print(f"!!! {RUNS_DIR} not found — is Drive mounted?")
    print("Run: from google.colab import drive; drive.mount('/content/drive')")
    sys.exit()

master = []
for sub in sorted(RUNS_DIR.iterdir()):
    if not sub.is_dir(): continue
    cfg_p, met_p = sub/"config.json", sub/"metrics.csv"
    if not cfg_p.is_file(): continue
    with open(cfg_p) as f: cfg = json.load(f)
    top1, top5 = "", ""
    if met_p.is_file():
        with open(met_p) as f:
            rows = list(csv.DictReader(f))
        test_rows = [r for r in rows if r.get("split","").lower()=="test"]
        final = test_rows[-1] if test_rows else (rows[-1] if rows else None)
        if final:
            for k, dst in (("top1","top1"),("top5","top5")):
                try:
                    v = float(final[k]); 
                    val = v*100 if v<=1.0 else v
                    if dst=="top1": top1 = val
                    else: top5 = val
                except: pass
    tags, init = classify(cfg)
    master.append({
        "run_name": sub.name, "tags": ";".join(tags), "seed": cfg.get("seed",""),
        "init": init, "region": cfg.get("region",""), "protocol": cfg.get("protocol",""),
        "probe_cond": cfg.get("probe_cond",""), "backbone_lr": cfg.get("lr_backbone",""),
        "head_lr": cfg.get("lr_head",""), "arc_m": cfg.get("arc_m",""),
        "freeze_backbone": int(bool(cfg.get("freeze_backbone",0))),
        "view_match": int(bool(cfg.get("knn_view_match",0))),
        "steps": cfg.get("steps",""), "batch_size": cfg.get("batch_size",""),
        "ckpt": cfg.get("ckpt",""), "top1": top1, "top5": top5,
    })

print(f"\n>>> Found {len(master)} runs in {RUNS_DIR}\n")

# --- write ---
cols = ["run_name","tags","seed","init","region","protocol","probe_cond",
        "backbone_lr","head_lr","arc_m","freeze_backbone","view_match",
        "steps","batch_size","ckpt","top1","top5"]

def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in cols})

write(OUT_DIR/"runs_master.csv", master)
print(f"  runs_master.csv  {len(master)} rows")

table_map = [("table02_main_results","table02"),("table03_ssl_contribution","table02"),
             ("table05_domain_comparison","table05"),("table06_lr_sweep","table06"),
             ("table08_graph_prior","table08"),("table09_spatial_only","table09"),
             ("table10_region_standard","table10"),("table11_data_scale","table11"),
             ("table12_data_scale_tests","table11")]
for name, tag in table_map:
    rows = [r for r in master if tag in r["tags"].split(";")]
    write(OUT_DIR/f"{name}.csv", rows)
    print(f"  {name}.csv  {len(rows)} rows")

print("\n>>> Done. Files in /content/logs/")