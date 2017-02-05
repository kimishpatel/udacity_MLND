#!/bin/bash

python SVHN_train_and_classify.py -f num_digits_networks/svhn_network.old.json -o "SVHN_results_num_extra_digits" -d 0

for i in `seq 1 5`:
do

output_file="SVHN_results_digit_extra_"$i
echo $output_file
for f in digit_${i}_networks/svhn_network.old.json
#for f in "${arr[@]}":
do
	echo "processing $f file to output $output_file on digit $i"
	python SVHN_train_and_classify.py -f $f -o $output_file -d $i
done

done
