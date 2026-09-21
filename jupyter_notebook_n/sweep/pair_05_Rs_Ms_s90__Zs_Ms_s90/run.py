# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py311-main
#     language: python
#     name: py311-main
# ---

# %% [markdown]
# # Sweep pair 05: Rs_Ms_s90 × Zs_Ms_s90
#
# Same-relationship pair as pair_04 (Rs_Ms_s61 × Zs_Ms_s61) at the s90
# snapshot instead of s61 -- lets you compare whether the size/metallicity
# inference relationship changes across snapshots.
#
# Runs the noise-model-only pipeline (7 cases: 5 mixed + 2 single-observable,
# 1500 epochs each, all 35 parameters, no focus params) for this observable
# pair, saves R2 heatmaps + a torch.save results.pt in ./results/.
#
# Hand-written to match generate_sweep.py's per-pair template (that generator
# only auto-picks its own random pairs; this pair was added by hand).

# %%
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.path.dirname(_HERE))  # sweep/
from _pair_pipeline import run_pair

# %%
run_pair(
    obs_a="Rs_Ms_s90",
    obs_b="Zs_Ms_s90",
    out_dir=os.path.join(_HERE, "results"),
    epochs=1500,
)
