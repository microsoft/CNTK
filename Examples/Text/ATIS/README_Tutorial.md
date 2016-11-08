
The ATIS corpus was converted to CTF format with these commands (tcsh):

# convert third column into the IOB format
cd Examples/Text/ATIS/Data
foreach f (atis.test atis.train)
 cat $f.tsv \
  | awk -F '\t' '{print $3}' $f.tsv \
  | awk '{if (NR == 4024) {$27 = "airfare";} print}' \
  | sed -e 's/'\''/ '\''/g' -e 's/o '\''clock/o'\''clock/g' \
        -e 's/ /  /g' -e 's/^/ /' -e 's/$/ /' \
        -e 's/ [^< ][^ ]* / O /g' \
        -e 's/<\([^\/]\)/B-\1/g' -e 's/B-\([^ ]*\)>  O/B-\1/g' \
        -e 's/B-\([^ ]*\)  O/B-\1  I-\1/g' \
        -e 's/I-\([^ ]*\)  O/I-\1  I-\1/g' \
        -e 's/I-\([^ ]*\)  O/I-\1  I-\1/g' \
        -e 's/<\/[^>]*>//g' \
        -e 's/  */ /g' -e 's/^ //' -e 's/ $//' \
  | paste $f.tsv - \
  | awk -F '\t' '{print "BOS "$1" EOS\t"$2"\tO "$4" O"}' \
  | sed -e 's/'\''/ '\''/g' -e 's/o '\''clock/o'\''clock/g' \
  > $f.txt
end

# get the vocabularies for each column
cat atis.train.txt atis.test.txt | awk -F '\t' '{print $1}' | tr ' ' '\n' | sort -u > query.wl
cat atis.train.txt atis.test.txt | awk -F '\t' '{print $2}' | tr ' ' '\n' | sort -u > intent.wl
cat atis.train.txt atis.test.txt | awk -F '\t' '{print $3}' | tr ' ' '\n' | sort -u > slots.wl

# create CTF file
foreach f (atis.test atis.train)
 python ../../../../Scripts/txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input $f.txt --output $f.ctf
end
