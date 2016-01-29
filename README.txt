This is a simple example of a character-level multi-layer RNN in CNTK, motivated by Andrej Karpathy's 
example (using Torch) in https://github.com/karpathy/char-rnn 

For simplicity, text is assumed to be in ASCII format (7-bit).
Non-printable characters are suppressed (null, delete, control characters, etc)

The first step is to convert the text input to 1-hot vectors. 
Use the ASCIITo1HotUCI script to generate UCI format input files:

	python ASCIITo1HotUCI.py <ascii-text-input-file> <uci-format-output-file>

Some sample input data is in the ./data directory




