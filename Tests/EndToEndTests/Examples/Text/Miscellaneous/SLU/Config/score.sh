# need to have outputs and labels
# in this case is output.rec.txt and output.lbl.txt

awk 'NF {str1=$1; getline < "output.rec.txt"; print str1" "$1 > "output.12.txt"}' feat.txt

awk '{str1=$1; str2=$2; getline < "output.lbl.txt"; print str1" NNP "$1 " "str2 > "output.all.txt"}' output.12.txt

cat output.all.txt | perl -ne '{chomp;s/\r//g;print $_,"\n";}' | perl conlleval.pl | head -10


# an example of feature and labels are respectively in 
# feat.txt and lbl.txt
