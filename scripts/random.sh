#!/bin/bash

python3 ../datahelper/sat_exact_count.py --id train-small --min_var 10 --max_var 30 --min_cl 20 --max_cl 50
python3 ../datahelper/sat_exact_count.py --id test-bigger --min_var 50 --max_var 75 --min_cl 50 --max_cl 100 --train False
python3 ../datahelper/sat_exact_count.py --id test-bigger-two --min_var 50 --max_var 75 --min_cl 130 --max_cl 150 --train False
python3 ../datahelper/sat_exact_count.py --id test-bigger-three --min_var 100 --max_var 150 --min_cl 50 --max_cl 100 --train False
python3 ../datahelper/sat_exact_count.py --id test-bigger-four --min_var 300 --max_var 450 --min_cl 200 --max_cl 350 --train False
