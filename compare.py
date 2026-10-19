"""Summarise runs/*.json into a table, including val loss at matched tokens and at matched wall-clock time."""
import json, glob, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "runs/*.json")))}
runs = {n: r for n, r in runs.items() if "eval_T256" in r}
known = ["gpt", "sghm", "sghm_nogate", "sghm_nomem", "hpt_xxs", "hpt", "hpt_s", "hpt_m", "hpt_l"]
order = [n for n in known if n in runs] + [n for n in runs if n not in known]


def val_at(r, key, lim):
    best = [row["val"] for row in r["log"] if "val" in row and row[key] <= lim]
    return best[-1] if best else float("nan")


common = min(r["tokens"] for r in runs.values())
common_t = min(r["train_time_s"] for r in runs.values())
print(f"{'model':10} {'params':>7} {'MFLOP/t':>7} {'tok/s':>7} {'TFLOPS':>6} {'tokens':>7} {'val@T256':>9} {'ppl':>6} "
      f"{'val@T1024':>9} {'>256':>6} {'repeat':>7} {'novel':>7} {f'val@{common / 1e6:.0f}M tok':>13} {f'val@{common_t:.0f}s':>9}")
for n in order:
    r = runs[n]
    e, e2 = r["eval_T256"], r["eval_T1024"]
    tps = r.get("steady_tok_per_s", r["tok_per_s"])
    fpt = r.get("flop_per_token", 6 * r["params"])
    print(f"{n:10} {r['params'] / 1e6:6.2f}M {fpt / 1e6:7.2f} {tps / 1e6:6.2f}M {tps * fpt / 1e12:6.1f} {r['tokens'] / 1e6:6.0f}M "
          f"{e['loss']:9.3f} {math.exp(e['loss']):6.1f} {e2['loss']:9.3f} {e2.get('loss_beyond_256', float('nan')):6.3f} "
          f"{e['repeat_content_loss']:7.3f} {e['novel_content_loss']:7.3f} {val_at(r, 'tokens', common):13.3f} "
          f"{val_at(r, 'time', common_t):9.3f}")
