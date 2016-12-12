/*
base64.cc - c++ source to a base64 reference encoder and decoder

This is part of the libb64 project, and has been placed in the public domain.
For details, see http://sourceforge.net/projects/libb64
*/

#include <b64/encode.h>
#include <b64/decode.h>

#include <iostream>
#include <fstream>
#include <string>

#include <stdlib.h>

// Function which prints the usage of this executable
void usage()
{
	std::cerr<< \
		"base64: Encodes and Decodes files using base64\n" \
		"Usage: base64 [-e|-d] [input] [output]\n" \
		"   Where [-e] will encode the input file into the output file,\n" \
		"         [-d] will decode the input file into the output file, and\n" \
		"         [input] and [output] are the input and output files, respectively.\n";
}
// Function which prints the usage of this executable, plus a short message
void usage(const std::string& message)
{
	usage();
	std::cerr<<"Incorrect invocation of base64:\n";
	std::cerr<<message<<std::endl;
}

int main(int argc, char** argv)
{
	// Quick check for valid arguments
	if (argc == 1)
	{
		usage();
		exit(-1);
	}
	if (argc != 4)
	{
		usage("Wrong number of arguments!");
		exit(-1);
	}
	
	// So far so good; try to open the input file
	std::string input = argv[2];
	// Note that we have to open the input in binary mode.
	// This is due to some operating systems not using binary mode by default.
	// Since we will most likely be dealing with binary files when encoding, we
	// have to be able to deal with zeros (and other invalid chars) in the input stream.
	std::ifstream instream(input.c_str(), std::ios_base::in | std::ios_base::binary);
	if (!instream.is_open())
	{
		usage("Could not open input file!");
		exit(-1);
	}
	
	// Now try to open the output file
	std::string output = argv[3];
	// Again, note that we have to open the ouput in binary mode.
	// Similiarly, we will most likely need to deal with zeros in the output stream when we
	// are decoding, and the output stream has to be able to use these invalid text chars.
	std::ofstream outstream(output.c_str(), std::ios_base::out | std::ios_base::binary);
	if (!outstream.is_open())
	{
		usage("Could not open output file!");
		exit(-1);
	}

	// determine whether we need to encode or decode:
	std::string choice = argv[1];
	if (choice == "-d")
	{
		base64::decoder D;
		D.decode(instream, outstream);
	}
	else if (choice == "-e")
	{
		base64::encoder E;
		E.encode(instream, outstream);
	}
	else
	{
		std::cout<<"["<<choice<<"]"<<std::endl;
		usage("Please specify -d or -e as first argument!");
	}

	return 0;
}

