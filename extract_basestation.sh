#!/bin/bash
bs=$1
file=$2
filename=$(echo "${file%%.*}")

output="${filename}_bs${bs}.csv"

awk -F, '{ print }' $file
#awk -F, '{ if ($2 == $bs || $3 == $bs)  print }'
#$file > $output
