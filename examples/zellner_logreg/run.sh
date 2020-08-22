#!/bin/bash

for ID in {1..3}
do
  for dnm in "adult" #"santa100K" "webspam"
  do
	   for graddiag in "False" "True"
				do
	     for i0 in "0.1" "1." "10."
						do
	       for f_rate in "0.0" "10.0" "20.0"
        do
	         for alg in "BPSVI" "BCORES" "SVI" # "PRIOR" "RAND"
          do
	           python3 main.py $alg $dnm $ID $graddiag $i0 $f_rate
          done
					   done
				  done
	   done
		done
done
