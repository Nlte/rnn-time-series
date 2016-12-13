#!/bin/bash

awk -F","  'BEGIN {OFS=","} { if ($1 ~ /^2013-01-0[1-7]/ ) print }' SET1V_01.CSV > exo1_week1.csv

awk  '{gsub(/-/,",");print}' data/exo1_bs2.csv > data/exo1_week12.csv
