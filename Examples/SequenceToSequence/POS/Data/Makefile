pos.train: train.txt.gz conll2cntk.pl
	zcat $< | ./conll2cntk.pl vocab label > $@

train.txt.gz:
	wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz

test.txt.gz:
	wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
