#!/bin/bash

triple="([[:digit:]]+[[:space:]]+){3}"
dateformat="([[:digit:]]|#){4}-([[:digit:]]|#){2}-([[:digit:]]|#){2}"
blankdate="####-##-##"

wikidata_format_errors()
{
    echo
    echo "wikidata errors:"
    grep -Ev --color "$triple$dateformat[[:space:]]$dateformat" wikidata12k/original/{train,test,valid}.txt # print offending lines
    grep -Evc --color "$triple$dateformat[[:space:]]$dateformat" wikidata12k/original/{train,test,valid}.txt # count
}

wikidata_timestamps_before()
{
    COUNT=0
    while IFS=$' \t\n' read -r head rel tail timefrom timeto ; do # split at " "
        year=${timefrom%%-*}

        if [[ $year != "####" ]] && (( $year<1709 )); then
            COUNT=$(($COUNT+1))
            #echo $head $rel $tail $timefrom $timeto
        fi
    done < wikidata12k/original/train.txt

    echo
    echo Number of timestamps in timefrom before 1709: $COUNT
}

no_blank_timestamps(){
    echo
    echo "Number of blankt timestamps in $1:"
    grep -Ec --color "$triple$blankdate.*" $1/original/{train,test,valid}.txt # count
}

wikidata_format_errors
wikidata_timestamps_before
no_blank_timestamps wikidata12k
no_blank_timestamps yago11k