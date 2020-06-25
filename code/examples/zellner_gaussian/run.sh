#!/bin/bash

for ID in {0..0}
do
    for alg in "SVI" "BCORES"  "BPSVI" #"GIGAO" "GIGAR" "RAND"
    do
			python3 main.py $alg $ID
    done
done
