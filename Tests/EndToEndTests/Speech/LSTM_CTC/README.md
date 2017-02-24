CNTK implementation of Connectionist Temporal Classification (CTC) is based on the paper by Alex Graves, etl. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks".  
These instructions are written in the context of speech recognition, yet remain relevant to handwriting recognition and other domains.
The criterion node for CTC training is *ForwardBackward*. This node is an abstraction for different types of training methods based on the forward/backward Viterbi-like pass. The node takes as the input the graph of labels, produced by the *LabelsToGraph* node that determines the exact forward/backward procedure. The *EditDistanceError* node is used for error evaluation. Here is [example CTC training configuration](https://github.com/Microsoft/CNTK/blob/master/Tests/EndToEndTests/Speech/LSTM_CTC/lstm.bs).

# Preparation of Training Data

Features are consumed with the HTKMLFDeserializer and labels with the TextDeserializer. To convert a traditional set of MLF/SCP files to the format suitable for CTC training, use [this python script](https://github.com/vmazalov/Scripts/blob/master/ctc_label_conversion.py). Assuming that the python script is placed into the CNTK directory with speech test data, example run to convert features is
<pre><code> %CNTK_path%\Tests\EndToEndTests\Speech\Data>python ctc_label_conversion.py --inputScpFile glob_0000.scp --inputMlfFile glob_0000.mlf --inputPhoneListFile state.list --outputScpFile ctc_glob_0000.scp --outputLabelFile ctc_glob_0000.mlf </code></pre>

There are several assumptions about the input data:

* The CTC blank token is expected to be the last label in the label sequence. For the referenced example, the blank token is a label with index 132.

*  The id of the blank token should be provided in the ForwardBackward node and as the tokenToIgnore in the EditDistanceError node.

* The label file (consumed by TextDeserializer) is expected to have the following format: 
  *seqId |l phoneId:[1|2]*,
where *seqId* is the id of the sequence/utterance to which given frame belongs, *phoneId* is index of the phone of given frame. *PhoneId* is followed by ":1" if this is not the first frame of the phone, or followed by ":2" if this is the first frame of the phone.
