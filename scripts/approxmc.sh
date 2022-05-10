#!/bin/bash

python3 ../datahelper/approxmc.py --folder test-bigger
python3 ../datahelper/approxmc.py --folder test-bigger-two
python3 ../datahelper/approxmc.py --folder test-bigger-three
python3 ../datahelper/approxmc.py --folder test-bigger-four

python3 ../datahelper/approxmc.py --folder dominating-set
python3 ../datahelper/approxmc.py --folder graph-coloring
python3 ../datahelper/approxmc.py --folder clique-detection