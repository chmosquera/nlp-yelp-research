#!/bin/sh


echo "Start Sentiment Calculator: "

ANALYZED="./analyzed/*"
RESULTS="./results/"
for file in $ANALYZED
do
    echo $($file | cut -f 1 -d '.')
    # echo $file
    # base=$(echo "$file" | cut -f 1 -d '.')
    # echo $base
    # python sentiment_calculator/SO_Calc.py -i $file -bo '${RESULTS}${base}.out' -ro '${RESULTS}{base}.rich'
done

echo "Done! Sentiment Calculation"
