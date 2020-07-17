#!/bin/bash

for ID in {1..1}
do
  # i. Transactions Dataset
	dnm="ds1"
	graddiag="False"
	i0="1."
	f_rate="20.0"

	for alg in  "BPSVI" "BCORES" "SVI"
  do
	  python3 main.py $alg $dnm $ID $graddiag $i0 $f_rate
  done
done
