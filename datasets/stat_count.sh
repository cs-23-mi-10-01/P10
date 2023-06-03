#!/bin/bash

triple="([[:digit:]]+[[:space:]]+){3}"
dateformat="([[:digit:]]|#){4}-([[:digit:]]|#){2}-([[:digit:]]|#){2}"

grep -Ev --color "^$triple$dateformat[[:space:]]$dateformat" wikidata12k/original/{train,test,valid}.txt # print offending lines
grep -Evc --color "^$triple$dateformat[[:space:]]$dateformat" wikidata12k/original/{train,test,valid}.txt # count

COUNT=0
while IFS=$' \t\n' read -r head rel tail timefrom timeto ; do # split at " "
    year=${timefrom%%-*}

    if [[ $year != "####" ]] && (( $year<1709 )); then
        COUNT=$(($COUNT+1))
        #echo $head $rel $tail $timefrom $timeto
    fi
done < wikidata12k/original/train.txt

echo Number of timestamps in timefrom before 1709: $COUNT