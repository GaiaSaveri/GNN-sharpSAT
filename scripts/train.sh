#!/bin/bash

python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id train-small
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id train-small
# FVGAT-VFNONE
python3 ../main.py --name bpgat_fv --attention_factor_var True --id train-small
# FVNONE-VFGAT
python3 ../main.py --name bpgat_vf --attention_var_factor True --id train-small
# FVMLP-VFGAT
python3 ../main.py --name bpnn_bpgat --mlp_factor_var True --attention_var_factor True --id train-small
# FVGAT-VFMLP
python3 ../main.py --name bpgat_bpnn --mlp_var_factor True --attention_factor_var True --id train-small
# BPGAT_VF
python3 ../main.py --name bpgat_damp_fv --attention_factor_var True --attention_var_factor True --mlp_damp_var_factor True --id train-small
# BPGAT_ALL
python3 ../main.py --name bpgat_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --mlp_damp_var_factor True --id train-small
# BPGAT_NONE
python3 ../main.py --name bpgat_no_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var False --mlp_damp_var_factor False --id train-small

# TS-BPNN
python3 ../main.py --name clique_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id clique-detection --n_epochs 501
python3 ../main.py --name dominating_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id dominating-set --n_epochs 501
python3 ../main.py --name coloring_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id graph-coloring --n_epochs 501
# TS-BPGAT
python3 ../main.py --name clique_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id clique-detection --n_epochs 501
python3 ../main.py --name dominating_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id dominating-set --n_epochs 501
python3 ../main.py --name coloring_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id graph-coloring --n_epochs 501