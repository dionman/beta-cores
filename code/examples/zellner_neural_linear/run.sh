#!/bin/bash

for tr in {0..0}
  do
		for f_rate in 0
		do
			 for beta in 0.1
				do
      for dnm in  "boston"
        do
        for alg in "SVI" #"RAND" #"BCORES"  #"BPSVI" "RAND"
          do
			          python3 main.py $dnm $alg $tr $f_rate $beta
							   done
								done
      done
    done
  done



## boston RAND 5.6219 -> 2.6498
##        SVI  3.7920 -> 2.8422
