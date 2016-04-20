from sys import argv

script, filename_in, filename_out = argv


file_in = open(filename_in, 'r')
file_out = open(filename_out, 'w')

for line in file_in.readlines():
	values = line.split( )
	label = values[0]
	dense_label = ['0'] * 10;
	dense_label[int(label)] = '1';
	file_out.write("|L " + " ".join(dense_label))
	file_out.write("\t")
	file_out.write("|F " + " ".join(values[1:]))
	file_out.write("\n")

file_in.close();
file_out.close();
	
