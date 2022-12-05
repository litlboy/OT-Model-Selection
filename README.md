# Brenier Criterion for Convex Transport Potential Selection

Code for the article [on model selection for convex OT models](https://openreview.net/forum?id=toleacrf7Hv). Training and parameter selection of three models: ICNNOT, Sinkhorn and SSNB.

![Using the semi-dual criterion, we can select the parameter whose associated potential (in full red) that best matches the ground truth (in blue).](https://github.com/[litlboy]/[OT_model_selection]/blob/[main]/sink_selection_1d.png?raw=true)

## Reproduce the experiments

From the command line, write for instance
'''bash
python Synth-XP/Lse/sinkhorn.py
'''
