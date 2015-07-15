# python scripts

'''
add silence ending to the begining and ending of a sentence 
the silence ending and begining symbol is </s>
example:
add_silence_ending('//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.txt')
'''
def add_silence_ending(fn, fnout):
	outfile = open(fnout, 'wt')
	with open(fn) as infile:
		for line in infile:
			line = line.strip()
			newline = '</s> ' + line + ' </s>'
			outfile.write(newline + '\n')

	outfile.close()


'''
create validation (first 100), test (last 100) and training data (remainning) split
example:
split_data_into_train_valid_test('//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.train.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.valid.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.test.txt')
'''
def split_data_into_train_valid_test(fn, fntrain, fnvalid, fntest):
	outfile_train = open(fntrain, 'wt')
	outfile_valid = open(fnvalid, 'wt')
	outfile_test = open(fntest, 'wt')

	# first get the line numbers
	totalln = 0
	with open(fn) as infile:
		for ln in infile:
			totalln += 1

	linenbr = 0
	with open(fn) as infile:
		for line in infile:
			if linenbr < 0.1 * totalln:
				outfile_valid.write(line)
			elif linenbr > 0.9 * totalln:
				outfile_test.write(line)
			else:
				outfile_train.write(line)

			linenbr += 1
	outfile_train.close()
	outfile_test.close()
	outfile_valid.close()

'''
convert to ascii file
example:
util.convert2ascii('//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/comments.cntk.ascii.txt')
'''
def convert2ascii(fn, fnout):
	import codecs
	of = open(fnout, 'wt')
	with open (fn) as infile:
		for line in infile:
			line = line.strip()
			if len(line) > 0:
				lineu = line.decode('utf8')
				of.write(lineu.encode("ASCII", 'ignore'))
				of.write('\n')

	of.close()

'''
remove agency 
util.removeagency('//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/news.ascii.txt', '//speechstore5/transient/kaishengy/data/newscomments/2015/03-23/news.ascii.noagency.txt')
'''
def removeagency(fn, fnout):
	import codecs
	of = open(fnout, 'wt')
	with open (fn) as infile:
		for line in infile:
			line = line.strip()
			agency_index = 0
			if ') -' in line:
				agency_index = line.find(') -')
				agency_index += 4
			if agency_index == 0 and ') ' in line:
				agency_index = line.find(') ')
				agency_index += 3

			nline = line[agency_index:]

			of.write(nline)
			of.write('\n')

	of.close()
