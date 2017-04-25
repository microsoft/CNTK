import numpy as np
import struct

def convert(info):
	in_image, in_label, out_file, line_count = info
	
	binfile = open(in_label , 'rb')
	labels = binfile.read()
	binfile.close()

	binfile = open(in_image, 'rb')
	images = binfile.read()
	binfile.close()
 
	label_index = 0
	image_index = 0;
	magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , images , image_index)
	image_index += struct.calcsize('>IIII')
	magic, num = struct.unpack_from('>II', labels, label_index)
	label_index += struct.calcsize('>II')

	output = open(out_file,'w')
	for i in range(line_count):
		label = struct.unpack_from('>B', labels, label_index)
		label = int(label[0])
		output.write(str(label));

		im = struct.unpack_from('>784B' ,images, image_index)
		im = np.array(im)
		for j in range(784): 
			output.write(' ' + str(im[j]))

		output.write('\n')
		image_index += struct.calcsize('>784B')
		label_index += struct.calcsize('>B')
	output.close()

a={'train':('train-images-idx3-ubyte','train-labels-idx1-ubyte','train.data', 60000),'test':('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 'test.data', 10000)}		
import sys
convert(a[sys.argv[1]])
