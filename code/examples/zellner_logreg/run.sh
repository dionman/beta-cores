#!/bin/bash

for ID in {1..1}
do
  # i. Transactions Dataset
	dnm="santa100K"
	stan_samples="True"
	samplediag="True"
	graddiag="False"
	i0="0.1"

	for alg in "BCORES" #"PRIOR" "RAND" "SVI" "BPSVI"
  do
	  python3 main.py $alg $dnm $ID $stan_samples $samplediag $graddiag $i0
  done
done
