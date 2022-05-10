#!/bin/bash

python3 ../datahelper/cnfgen.py --n_problems 250 --id dominating-set --kind train --cnfgen_cmd domset --n 15 --p 0.6
python3 ../datahelper/cnfgen.py --n_problems 50 --id dominating-set --kind validation --cnfgen_cmd domset --n 15 --p 0.6
python3 ../datahelper/cnfgen.py --n_problems 100 --id dominating-set --kind test --cnfgen_cmd domset --n 15 --p 0.6

python3 ../datahelper/cnfgen.py --n_problems 250 --id graph-coloring --kind train --cnfgen_cmd kcolor --n 10 --p 0.6
python3 ../datahelper/cnfgen.py --n_problems 50 --id graph-coloring --kind validation --cnfgen_cmd kcolor --n 10 --p 0.6
python3 ../datahelper/cnfgen.py --n_problems 100 --id graph-coloring --kind test --cnfgen_cmd kcolor --n 10 --p 0.6

python3 ../datahelper/cnfgen.py --n_problems 250 --id clique-detection --kind train --cnfgen_cmd kclique --n 15 --p 0.5
python3 ../datahelper/cnfgen.py --n_problems 50 --id clique-detection --kind validation --cnfgen_cmd kclique --n 15 --p 0.5
python3 ../datahelper/cnfgen.py --n_problems 100 --id clique-detection --kind test --cnfgen_cmd kclique --n 15 --p 0.5