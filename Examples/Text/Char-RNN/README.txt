This is a simple example of a character-level multi-layer RNN in CNTK, motivated by Andrej Karpathy's 
example (using Torch) in https://github.com/karpathy/char-rnn

For simplicity, text is assumed to be in ASCII format (7-bit).
Non-printable characters are suppressed (null, delete, control characters, etc)
This is not generally suitable for extended character set or Unicode text!

This example is intended to be small enough to run on a non-GPU environment for getting familiar with CNTK, 
but also illustrate some of the steps for larger tasks.

Steps:

1.  The first step is to convert the text input to 1-hot vectors. A single component of a 1-hot vector
    is 1, and corresponds to the presence of a single input symbol, with all other components being 0.
    For this example, we limit the number of possible input symbols by requiring the text to be 7-bit ASCII.
    In general, any finite symbol set can be represented as 1-hot vectors, but may be impractically large.
    
    Use the ASCIITo1HotUCI script to generate UCI format input files:

	    python ASCIITo1HotUCI.py <ascii-text-input-file> <uci-format-output-file>

    The output will have one row of text for each letter of input. "UCI format" is mostly just 
    space-delimited numerical text. UCI format files can also include a label at the beginning or end 
    of each row. A sample of the output from ASCIITo1HotUCI.py:

	e    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 

    Some sample input data is in the ./data directory. You can try your own text files as well.


2.  Train the model by running CNTK with the Char-RNN.cntk config file. The command will look like this:

        CNTK.exe -configFile=..\Char-RNN.cntk 
 

 3. Use the model to generate some output text

        <need to write this>







