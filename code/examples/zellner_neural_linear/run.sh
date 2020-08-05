#!/bin/bash

for tr in {0..0}
  do
		for f_rate in 0.1 0
		do
			 for beta in 0.9
				do
      for dnm in  "boston"
        do
        for alg in "BCORES" #"SVI" #"BPSVI" "RAND"
          do
			          python3 main.py $dnm $alg $tr $f_rate $beta
							   done
								done
      done
    done
  done
