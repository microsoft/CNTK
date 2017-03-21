# Build Language Understanding Models with CNTK

This example demonstrates how to build a language understanding model with CNTK using the ATIS data set. The recipes in this example are in BrainScript. For Python click [here](https://github.com/Microsoft/CNTK/blob/master/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py).  This example uses the CNTKTextFormatReader, sparse contextual feature vectors and sparse label input.

The Air travel information system (ATIS) corpus is used for training and testing.
## Download the example
The data and configuration is checked in to github. You can get it by command:
B
`git clone https://github.com/Microsoft/cntk`

The example is under folder: 
`<cntk_root>\Examples\LanguageUnderstanding\ATIS\BrainScript`

## Data File Format
There are four files under `data` sub-folder

|Files                  |Content |
|:----------------------|:--------|
|ATIS.train.cntk.sparse |featurized training data set  
|ATIS.test.cntk.sparse  |featurized test data set 
|ATIS.vocab             |all words extracted from training data. Vocab size: 944 
|ATIS.labels            |all semantic labels extracted from training data. Total labels: 127 

We preprocess ATIS data by converting words into word indexes, and labels into label IDs in order to use 
[CNTKTextFormatReader](https://github.com/Microsoft/CNTK/wiki/BrainScript-CNTKTextFormat-Reader). You can use any 
script/tool to preprocess your text data files. In this example, data is already preprocessed.

The last two files ATIS.vocab and ATIS.labels are not really required to run the example. They are included for evaluation and debugging purpose. 
E.g. they can be used to convert .sparse files back to original text files. 

To understand the data format (two .sparse files), let's start with a sample sentence:
```
BOS i would like to find a flight from charlotte to Las Vegas that makes a stop in St. Louis EOS
```
it is converted into the following text:
```
1    |PW 1:1     |CW 1:1     |NW 12:1    |L 126:1
1    |PW 1:1     |CW 12:1    |NW 39:1    |L 126:1
1    |PW 12:1    |CW 39:1    |NW 28:1    |L 126:1
1    |PW 39:1    |CW 28:1    |NW 3:1     |L 126:1
1    |PW 28:1    |CW 3:1     |NW 86:1    |L 126:1
1    |PW 3:1     |CW 86:1    |NW 15:1    |L 126:1
1    |PW 86:1    |CW 15:1    |NW 10:1    |L 126:1
1    |PW 15:1    |CW 10:1    |NW 4:1     |L 126:1
1    |PW 10:1    |CW 4:1     |NW 101:1   |L 126:1
1    |PW 4:1     |CW 101:1   |NW 3:1     |L 48:1
1    |PW 101:1   |CW 3:1     |NW 92:1    |L 126:1
1    |PW 3:1     |CW 92:1    |NW 90:1    |L 78:1
1    |PW 92:1    |CW 90:1    |NW 33:1    |L 123:1
1    |PW 90:1    |CW 33:1    |NW 338:1   |L 126:1
1    |PW 33:1    |CW 338:1   |NW 15:1    |L 126:1
1    |PW 338:1   |CW 15:1    |NW 132:1   |L 126:1
1    |PW 15:1    |CW 132:1   |NW 17:1    |L 126:1
1    |PW 132:1   |CW 17:1    |NW 72:1    |L 126:1
1    |PW 17:1    |CW 72:1    |NW 144:1   |L 71:1
1    |PW 72:1    |CW 144:1   |NW 2:1     |L 119:1
1    |PW 144:1   |CW 2:1     |NW 2:1     |L 126:1
```
where the first column identifies the sequence (sentence) ID, which is the same for all words of the same sentence. There are four input streams: PW, CW, NW, L. 
The input "PW" represents the previous word ID, "CW" for current word, and "NW" for next word. Input name "L" is for labels. The input names can be anything you 
like and you can add more input as needed, e.g. words in a bigger window.

Words "BOS" and "EOS" denote beginning of sentence and end of sentences respectively.

Each line above represents one sample (word). E.g. the meaning of this line: `1	|PW 4:1	|CW 101:1	|NW 3:1	|L 48:1`:
* the sequence ID is 1
* the current word is "charlotte" whose word ID is 101
* the previous word is "from" whose ID is 4
* the next word is "to" whose ID is 3
* the semantic label is "B-fromloc.city_name" whose label Id is 48.

All word IDs, label IDs and corresponding words and labels are stored in ATIS.vocab and ATIS.labels.

## CNTK Configuration

In this example, we use BrainScript to create one-layer LSTM with embedding for slot tagging. The consolidated config file is ATIS.cntk. One can check the file (with some comments) 
for details, especially how the reader is configured in ATIS.cntk.

    reader=[
        readerType = "CNTKTextFormatReader" 
        file = "$DataDir$/ATIS.train.cntk.sparse" 

        miniBatchMode = "partial" 
        randomize = true
        input = [
            featuresPW = [ 
                alias = "PW"    # previous word
                dim = $wordCount$ 
                format = "sparse" 
            ] 
            featuresCW = [ 
                alias = "CW"    # current word
                dim = $wordCount$ 
                format = "sparse" 
            ]
            featuresNW = [ 
                alias = "NW"    # next word
                dim = $wordCount$ 
                format = "sparse" 
            ]
            
            labels = [ 
                alias = "L"     # label
                dim = $labelCount$
                format = "sparse" 
            ] 
        ]
    ]  

The above section tells CNTK to use CNTKTextFormatReader to read data from the file "$DataDir/ATIS.train.cntk.sparse". The same input names (PW, CW, NW, L) are used to refer inputs (features and labels) provided in data files. The input is read into different 
feature vectors: featuresPW, featuresCW, featuresNW and labels. These vectors are later used to build LSTM node with BrainScript as follows. 
```
        featuresPW = Input(inputDim)
        featuresCW = Input(inputDim)
        featuresNW = Input(inputDim)
        features = RowStack(featuresPW : featuresCW : featuresNW)
        labels=Input(labelDim, tag="label")
        # embedding layer
        emb = LearnableParameter(embDim, featDim)
        featEmbedded = Times(emb, features)
        # build the LSTM stack
        lstmDims[i:0..maxLayer] = hiddenDim
        NoAuxInputHook (input, lstmState) = BS.Constants.None
        lstmStack = BS.RNNs.RecurrentLSTMPStack (lstmDims, 
            cellDims=lstmDims,
            featEmbedded, 
            inputDim=embDim,
            previousHook=BS.RNNs.PreviousHC,
            augmentInputHook=BS.RNNs.NoAuxInputHook, 
            augmentInputDim=0,
            enableSelfStabilization=false)
        lstmOutputLayer = Length (lstmStack)-1
        LSTMoutput = lstmStack[lstmOutputLayer].h

```
A few other notes about the config:
- it is important to specify the format is "sparse".
- the gradUpdateType is set FSAdaGrad. This setting reports better model accuracy comparing any other update methods.
- multiple LSTM layers can be used by changing the value of maxLayer.

Three commands are configured: Train, Output and Test. The command "Train" is used to train a model, "Output" is used to evaluate the model against a test set and store
the model output, and the command "Test" is to calculate the model's accuracy.

## Run the example

One can run the example locally or on Philly (for Microsoft internal users). 

To run locally,

```sh
> mkdir work              # the default work_dir
> open ATIS.cntk and update the value of deviceId: -1 for CPU, auto for GPU
> cntk configFile=ATIS.cntk
```

By default, the maxEpochs is set to 1 to save training time. One can change it to larger value such as 20 in order to get a good model accuracy. 
Depends on GPU, it normally takes about 20 minutes to run 20 epochs on single GPU. The slot F1 score should be around 94 with 20 epochs.

**For Microsoft users only**, to run the job on Philly:
- first upload data folder to philly cloud. e.g. `\\storage.gcr.philly.selfhost.corp.microsoft.com\pnrsy\<your_alias>\ATIS `
- update the config file to philly cloud, e.g. `\\storage.gcr.philly.selfhost.corp.microsoft.com\pnrsy_scratch\<your_alias>\ATIS`
- go to http://philly/ to create a new job by specifying data folder and config file, and start the job.

More details about Philly, including how to upload data to Philly and start jobs, can be found [here](https://microsoft.sharepoint.com/teams/ATISG/SitePages/Philly%20Users%20Guide.aspx)
