# Build Neural Character Language Models with CNTK

This example demonstrates how to build a neural character language model with CNTK using regular plaintext data fed in using the numpy interface.

A neural language model uses a recurrent neural network to predict words (or characters) with a richer context than traditional n-gram models allow. In this implementation, a character is run through an LSTM and the output is then put through a fully-connected layer to predict the next output character. The model can learn to be extremely expressive as the context is progressively built-up with each letter run through the RNN. For even more expressiveness, we allow a stack of LSTMs where the output of each layer is put through the next layer as its input.

This example is inspired by Andrej Karpathy's blog post "The Unreasonable Effectiveness of Recurrent Neural Networks" at http://karpathy.github.io/2015/05/21/rnn-effectiveness/ and his accompanying code at https://github.com/karpathy/char-rnn. This example allows you to achieve similar results to those displayed in Karpathy's blog, but with the packed-sequence training efficiency that CNTK allows.

## HOWTO

Set the `hidden_dim` and `num_layers` to values that match the complexity/size of your data. To learn a model, simply call `train_lm(your_data)` where `your_data` is a plaintext file containing your training data. Once you have a model that you're happy with (the code is currently configured to save a new model at the end of each epoch [i.e. each pass over the full training data]), then call the function `load_and_sample` as in the following example:

`load_and_sample("models/shakespeare_epoch19.dnn", "tinyshakespeare.txt.vocab", prime_text=text, use_hardmax=False, length=100, temperature=0.95)`

In the above, we pass in the model saved in `models/shakespeare_epoch19.dnn` (i.e. the saved model after training for 20 epochs), the vocab `tinyshakespeare.txt.vocab` (which is automatically created from `tinyshakespeare.txt` when you train a model with that training data), the prime-text `text` (which will run some priming text through the model before sampling from it), `use_hardmax` set to `False` meaning that there will be some sampling instead of just always taking the most likely predicting from the model, the `length` of the sample you wish to generate (in characters, including the prime-text), and finally, the `temperature` where `1.0` means use the actual probabilities predicted by the model, and lower numbers flatten the distribution so that the samples will be less like the learned model but more "creative".

Have fun!

## Examples

Using the tiny-Shakespeare data (included) with a 2-layer LSTM and hidden-dim of 256:

```
KING up:
low to it; for he's mistress that I might see,
I spurn them in good words.

BEVIS:
Then sport what!
Madam, the vein o' the ill highest of the hide.

KING JOHN:
Fie, Henry, if thou be my reverend courage,
Whose two spurs poorer in or partless,
Yet riveted by his eld, il execution,
Lukess undout sound teach; four wives do sworn,
As with this carf's--God woo, to this! what tends?
Till this unlewn bushes are but fourteen,
Or sitter on our pyn; and on my better
A drum and fitness of my bearing
```

Using the 20-newsgroup data with a 2-layer LSTM and hidden-dim of 256:

```
Newsgroups: ut.whuhroel
Date: 20 Apr 93 15:03:08 GMT
Lines: 19


> I am annoying.  This way to the tapes of principle about everyone like validity must do you.

Dick. Good chunkan

mjson@austin.ibm.com
>-->

Xref: cantaloupe.srv.cs.cmu.edu comp.sys.ibm.pc.hardware:61655 alt.sockha.mangers.36784558 talk.religion.misc:99703
comp.os.ms-windows.misc:9720 comp.aiz.dack-orit:15@stp.noscne.com>
Organization: IBR, Aucidore.
Reply-To: jjlu@llo@minta.UU.NOT
>>Adutions a road-Tell 9:20 AX   10516 PHKRS

>BA
```