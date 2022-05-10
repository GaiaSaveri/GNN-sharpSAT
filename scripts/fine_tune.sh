#!/bin/bash

python3 ../main.py --name bpgat_dominating --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id dominating-set --restore_train True --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --n_epochs 751
python3 ../main.py --name bpnn_dominating --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id dominating-set --restore_train True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --n_epochs 751

python3 ../main.py --name bpgat_clique --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id clique-detection --restore_train True --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --n_epochs 751
python3 ../main.py --name bpnn_clique --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id clique-detection --restore_train True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --n_epochs 751

python3 ../main.py --name bpgat_coloring --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --id graph-coloring --restore_train True --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --n_epochs 751
python3 ../main.py --name bpnn_coloring --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --id graph-coloring --restore_train True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --n_epochs 751