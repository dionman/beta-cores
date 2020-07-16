#!/bin/bash

for ID in {1..1}
do
  # i. Transactions Dataset
	dnm="ds1"
	graddiag="True"
	i0="1."
	f_rate="0.0"

	for alg in  "RAND" "PRIOR" "BPSVI" "BCORES" "SVI"
  do
	  python3 main.py $alg $dnm $ID $graddiag $i0 $f_rate
  done
done
