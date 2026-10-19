"""Summarise runs/*.json into a table, including val loss at matched token counts."""
import json, glob, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "runs/*.json")))}
order = [n for n in ["gpt", "sghm", "sghm_nogate", "sghm_dense", "sghm_nomem"] if n in runs]


def val_at_tokens(r, tok):
    pts = [(row["tokens"], row["val"]) for row in r["log"] if "val" in row]
    best = [v for t, v in pts if t <= tok]
    return best[-1] if best else float("nan")


common = min(r["tokens"] for r in runs.values())
print(f"{'model':12} {'params':>7} {'tok/s':>7} {'tokens':>7} {'val@T256':>9} {'ppl':>6} {'val@T1024':>9} "
      f"{'>256':>6} {'repeat':>7} {'novel':>7} {f'val@{common / 1e6:.1f}M tok':>14} {'write%':>7}")
for n in order:
    r = runs[n]
    e, e2 = r["eval_T256"], r["eval_T1024"]
    wf = r["log"][-1].get("write_frac")
    print(f"{n:12} {r['params'] / 1e6:6.2f}M {r['tok_per_s'] / 1e3:6.1f}k {r['tokens'] / 1e6:6.1f}M {e['loss']:9.3f} "
          f"{math.exp(e['loss']):6.1f} {e2['loss']:9.3f} {e2.get('loss_beyond_256', float('nan')):6.3f} "
          f"{e['repeat_content_loss']:7.3f} {e['novel_content_loss']:7.3f} {val_at_tokens(r, common):14.3f} "
          f"{'' if wf is None else f'{100 * wf:6.1f}%':>7}")
