#!/bin/bash

for tr in {0..0}
  do
  for dnm in  "boston" 
    do
    for alg in "PRIOR" #"SVI" #"BPSVI" "BCORES" "GIGAO" "GIGAR" "RAND"
      do
			   python3 main.py $dnm $alg $tr
    done
  done
done
