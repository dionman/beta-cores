#!/bin/bash

fldr_figs='figs'
fldr_res='results'
for beta in  "0.01"
do
for i0 in "1.0"
do
for f_rate in "0"
do
for graddiag in "False"
do
for dnm in 'adult'
  do
    #echo $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
    python3 plot.py $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
  done
done
done
done
done


'''
fldr_figs='figs'
fldr_res='results'
for beta in "0.01" "0.01" "0.5" "0.9"
do
for i0 in "0.1" "1." "10."
do
for f_rate in "0" "15" "30"
do
for graddiag in "False" "True"
do
for dnm in 'webspam' 'adult' 'santa100K'
  do
    #echo $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
    python3 plot.py $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
  done
done
done
done
done
'''

'''
str(beta)+'_'+str(i0)+'_'+str(f_rate)+'_'+str(graddiag)

adult0.5_0.1_15_False_ACCvssz.png
adult0.5_1.0_30_False_ACCvssz.png
adult0.9_1.0_30_False_ACCvssz.png


santa100K0.01_10.0_15_False_ACCvssz.png
santa100K0.01_10.0_30_False_ACCvssz.png
santa100K0.5_10.0_15_True_ACCvssz.png
santa100K0.5_10.0_30_True_ACCvssz.png
'''
