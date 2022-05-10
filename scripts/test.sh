#!/bin/bash

python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=1000_info.pt --id test-bigger-three --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --id test-bigger-four --train False
python3 ../main.py --name dominating_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/dominating_bpnn/dominating_bpnn_epoch=500_info.pt --id dominating-set --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --id dominating-set --train False
python3 ../main.py --name bpnn_dominating --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn_dominating/bpnn_dominating_epoch=750_info.pt --id dominating-set --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --id graph-coloring --train False
python3 ../main.py --name coloring_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/coloring_bpnn/coloring_bpnn_epoch=500_info.pt --id graph-coloring --train False
python3 ../main.py --name bpnn_coloring --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn_coloring/bpnn_coloring_epoch=750_info.pt --id graph-coloring --train False
python3 ../main.py --name clique_bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/clique_bpnn/clique_bpnn_epoch=500_info.pt --id clique-detection --train False
python3 ../main.py --name bpnn --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn/bpnn_epoch=500_info.pt --id clique-detection --train False
python3 ../main.py --name bpnn_clique --mlp_factor_var True --mlp_var_factor True --mlp_damp_factor_var True --restore_file ../run/bpnn_clique/bpnn_clique_epoch=750_info.pt --id clique-detection --train False


python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=1000_info.pt --id test-bigger-three --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=1000_info.pt --id test-bigger-four --train False
python3 ../main.py --name dominating_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/dominating_bpgat/dominating_bpgat_epoch=500_info.pt --id dominating-set --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --id dominating-set --train False
python3 ../main.py --name bpgat_dominating --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat_dominating/bpgat_dominating_epoch=750_info.pt --id dominating-set --train False
python3 ../main.py --name coloring_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/coloring_bpgat/coloring_bpgat_epoch=500_info.pt --id graph-coloring --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --id graph-coloring --train False
python3 ../main.py --name bpgat_coloring --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat_coloring/bpgat_coloring_epoch=750_info.pt --id graph-coloring --train False
python3 ../main.py --name clique_bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/clique_bpgat/clique_bpgat_epoch=500_info.pt --id clique-detection --train False
python3 ../main.py --name bpgat --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat/bpgat_epoch=500_info.pt --id clique-detection --train False
python3 ../main.py --name bpgat_clique --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True  --restore_file ../run/bpgat_clique/bpgat_clique_epoch=750_info.pt --id clique-detection --train False


python3 ../main.py --name bpgat_fv --attention_factor_var True --restore_file ../run/bpgat_fv/bpgat_fv_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_fv --attention_factor_var True --restore_file ../run/bpgat_fv/bpgat_fv_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_fv --attention_factor_var True --restore_file ../run/bpgat_fv/bpgat_fv_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpgat_vf --attention_var_factor True --restore_file ../run/bpgat_vf/bpgat_vf_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_vf --attention_var_factor True --restore_file ../run/bpgat_vf/bpgat_vf_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_vf --attention_var_factor True --restore_file ../run/bpgat_vf/bpgat_vf_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpgat_damp_fv --attention_factor_var True --attention_var_factor True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp_fv/bpgat_damp_fv_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_damp_fv --attention_factor_var True --attention_var_factor True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp_fv/bpgat_damp_fv_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_damp_fv --attention_factor_var True --attention_var_factor True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp_fv/bpgat_damp_fv_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpgat_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp/bpgat_damp_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp/bpgat_damp_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var True --mlp_damp_var_factor True --restore_file ../run/bpgat_damp/bpgat_damp_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpgat_no_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var False --mlp_damp_var_factor False --restore_file ../run/bpgat_no_damp/bpgat_no_damp_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_no_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var False --mlp_damp_var_factor False --restore_file ../run/bpgat_no_damp/bpgat_no_damp_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_no_damp --attention_factor_var True --attention_var_factor True --mlp_damp_factor_var False --mlp_damp_var_factor False --restore_file ../run/bpgat_no_damp/bpgat_no_damp_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpnn_bpgat --mlp_factor_var True --attention_var_factor True --restore_file ../run/bpnn_bpgat/bpnn_bpgat_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpnn_bpgat --mlp_factor_var True --attention_var_factor True --restore_file ../run/bpnn_bpgat/bpnn_bpgat_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpnn_bpgat --mlp_factor_var True --attention_var_factor True --restore_file ../run/bpnn_bpgat/bpnn_bpgat_epoch=1000_info.pt --id test-bigger-three --train False

python3 ../main.py --name bpgat_bpnn --mlp_var_factor True --attention_factor_var True --restore_file ../run/bpgat_bpnn/bpgat_bpnn_epoch=1000_info.pt --id test-bigger --train False
python3 ../main.py --name bpgat_bpnn --mlp_var_factor True --attention_factor_var True --restore_file ../run/bpgat_bpnn/bpgat_bpnn_epoch=1000_info.pt --id test-bigger-two --train False
python3 ../main.py --name bpgat_bpnn --mlp_var_factor True --attention_factor_var True --restore_file ../run/bpgat_bpnn/bpgat_bpnn_epoch=1000_info.pt --id test-bigger-three --train False