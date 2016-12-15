#!/bin/bash
bs=$1
file=$2
filename=$(echo "${file%%.*}")

output="${filename}_bs${bs}.csv"

awk -F","  'BEGIN {OFS=","} { if ($2 == "2" || $3 == "2") print }' $file > $output
#awk -F","  'BEGIN {OFS=","} { if ($1 ~ /^2013-01-0[1-7]/ ) print }'
#awk -F","  'BEGIN {OFS=","} { if ($2 == "2" || $3 == "2")  print }' $file > $output
#awk -F, '{ if ($2 == $bs || $3 == $bs)  print }'
#$file > $output
