This directory contains experiments for grapheme-to-phoneme experiments reported in the following paper
K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme to phoneme conversion"
submitted to Interspeech 2015


encoder-decoder LSTM : scripts/s36.noreg.log
best performing systemis
s36noregrnds2sencoderh500c500decoderh500c500mb100mpdlr01layers2

unidirectional LSTM : scripts/s23.unidirectional.log

bidirectional LSTM: scripts/s30.bidirectional.log

-----------------------
to score: 

suppose the G2P results are in xxx.output
in Python. use the following commands


import const as cn
import score

outputfn='//speechstore5/transient/kaishengy/exp/lts/result/reps30bilstm/test_bw1_iter35/output.rec.txt'
outputfn='//speechstore5/transient/kaishengy/exp/lts/result/reps23mb100fw6/test_bw1_iter34/output.rec.txt'
outputfn='//speechstore5/transient/kaishengy/exp/lts/result/reprs36noregrnds2sencoderh500c500decoderh500c500mb100mpdlr01layers2/test_bw1_iter43/output.rec.txt'
outputfn='//speechstore5/transient/kaishengy/exp/lts/result/s30rndjointconditionalbilstmn300n300n300/test_bw1_iter35/output.rec.txt'
outputfn='//speechstore5/transient/kaishengy/exp/lts/result/s36noregrnds2sencoderh500c500decoderh500c500mb100mpdlr01layers2/test_bw1_iter43/output.rec.txt'
lexicon = {}
score.ReadPronunciations(cn._TEST_PRON, lexicon)
score.BleuScore(lexicon,
    cn._TEST_FN,
    outputfn,
    False)

score.CORRECT_RATE(lexicon,
    cn._TEST_FN,
    outputfn,
    False)
