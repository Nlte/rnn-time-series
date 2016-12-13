#!/bin/bash

awk -F"," 'BEGIN {OFS=","} FNR==NR{a[$1]=$3; b[$1]=$4;next}{print $1,$2,$3,$4,a[$5],b[$5],a[$6],b[$6],$7,$8}' SITE_ARR_LONLAT.CSV exo1_week1.csv
