#!/bin/bash

for ID in {1..1}
do
    for alg in "BCORES" # "SVI" "BPSVI" "GIGAO" "GIGAR" "RAND"
    do
			python3 main.py $alg $ID
    done
done

