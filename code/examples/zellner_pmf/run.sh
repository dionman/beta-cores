#!/bin/bash

for ID in {0..0}
do
    for alg in "SVI" #"BPSVI" "BCORES" "GIGAO" "GIGAR" "RAND"
    do
			python3 main.py $alg $ID
    done
done
