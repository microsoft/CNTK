# Build language understanding model from ATIS using CNTKTextFormatReader

This example demonstrates how to use build language understanding model with CNTK using ATIS data set. This example is similar to 
[SLU example](https://github.com/Microsoft/CNTK/tree/master/Examples/Text/Miscellaneous/SLU). They are different in that
  - CNTKTextFormatReader is used here, instead of LUSequenceReader
  - With CNTKTextFormatReader, the input format is much more flexible. In the example setting, sparse contextual feature vectors are explored
  - Sparse label input is used.

The Air travel information system (ATIS) corpus is used for training and testing. No development set is provided as it is not available.  
## Download the example
The data and configuration is checked in to github. You can get it by
`git clone https://github.com/Microsoft/cntk`

The example is under folder: 
`<cntk_root>\Examples\Text\atis`

## Data File Format
There are four files under `data` sub-folder
| Files  | Content
|:---------|:---|
|atis.train.cntk.sparse | featurized training data set
|atis.test.cntk.sparse  | featurized test data set
|atis.vocab             | all words extracted from training data. Vocab size: 944
|atis.labels            | all semantic labels extracted from training data. Total labels: 127

We preprocess ATIS data by converting words into word indexes, and labels into label IDs in order to use 
[CNTKTextFormatReader](https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader). You can use any 
script/tool to preprocess your text data files. In this example, data is already preprocessed.

The last two files atis.vocab and atis.labels are not really required to run the example. They are included for evaluation and debugging purpose. 
E.g. they can be used to convert .sparse files back to original text files. 

To understand the data format (two .sparse files), let's start with a sample sentence:
```
BOS i would like to find a flight from charlotte to Las Vegas that makes a stop in St. Louis EOS
```
it is converted into the following format:
```
1	|PW 1:1	|CW 1:1	|NW 12:1	|L 126:1
1	|PW 1:1	|CW 12:1	|NW 39:1	|L 126:1
1	|PW 12:1	|CW 39:1	|NW 28:1	|L 126:1
1	|PW 39:1	|CW 28:1	|NW 3:1	|L 126:1
1	|PW 28:1	|CW 3:1	|NW 86:1	|L 126:1
1	|PW 3:1	|CW 86:1	|NW 15:1	|L 126:1
1	|PW 86:1	|CW 15:1	|NW 10:1	|L 126:1
1	|PW 15:1	|CW 10:1	|NW 4:1	|L 126:1
1	|PW 10:1	|CW 4:1	|NW 101:1	|L 126:1
1	|PW 4:1	|CW 101:1	|NW 3:1	|L 48:1
1	|PW 101:1	|CW 3:1	|NW 92:1	|L 126:1
1	|PW 3:1	|CW 92:1	|NW 90:1	|L 78:1
1	|PW 92:1	|CW 90:1	|NW 33:1	|L 123:1
1	|PW 90:1	|CW 33:1	|NW 338:1	|L 126:1
1	|PW 33:1	|CW 338:1	|NW 15:1	|L 126:1
1	|PW 338:1	|CW 15:1	|NW 132:1	|L 126:1
1	|PW 15:1	|CW 132:1	|NW 17:1	|L 126:1
1	|PW 132:1	|CW 17:1	|NW 72:1	|L 126:1
1	|PW 17:1	|CW 72:1	|NW 144:1	|L 71:1
1	|PW 72:1	|CW 144:1	|NW 2:1	|L 119:1
1	|PW 144:1	|CW 2:1	|NW 2:1	|L 126:1
```
where the first column identifies the sequence (sentence) ID. Samples with the same sequence ID make a sentence. There are four input streams: PW, CW, NW, L. 
The input "PW" represents previous word ID, "CW" for current word, and "NW" for next word. Input name "L" is for labels. The input names can be anything you 
like and your can add more input as needed, e.g. words in a bigger window.

Words "BOS" and "EOS" denote beginning of sentence and end of sentences respectively.

Each line above represents one sample (word). E.g. the meaning of this line: `1	|PW 4:1	|CW 101:1	|NW 3:1	|L 48:1`:
* the current word is "charlotte" whose word ID is 101
* the previous word is "from" whose ID is 4
* the next word is "to" whose ID is 3
* the semantic label is "B-fromloc.city_name" whose label Id is 48.

All word IDs and label IDs are stored in atis.vocab and atis.labels.

## CNTK Configuration

In this example, we use BrainScript to create one-layer LSTM for slot tagging. The consolidated config file is atis.cntk. One can check the file (with some comments) 
for details, especially how the reader is configured in atis.cntk.

```
    reader=[
        readerType = "CNTKTextFormatReader" 
        file = "$DataDir$/atis.train.cntk.sparse" 

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
```
Note that the same input names (PW, CW, NW, L) are used to refer inputs (features and labels) provided in data files. The input is read into different 
feature vectors: featuresPW, featuresCW, featuresNW and labels. These vectors are later used to build LSTM node with BrainScript as follows. 
```
        featuresPW = Input(inputDim)
        featuresCW = Input(inputDim)
        featuresNW = Input(inputDim)
        features = RowStack(featuresPW : featuresCW : featuresNW)
        labels=Input(labelDim, tag="label")
        emb = LearnableParameter(embDim, featDim)
        featLookuped = Times(emb, features)
        # a single layer is used in this example
        LSTMoutput = ForwardLSTMComponent(embDim, hiddenDim, featLookuped, initScale, initBias)
```
A few other notes about the config:
- it is important to specify the format is "sparse".
- the gradUpdateType is set FSAdaGrad. This setting reported better model accuracy comparing any other update methods.
- multiple layers (commented out) can be added but it may not always perform better.

## Run the example

One can run the example locally or on Philly. 
To run locally,

```sh
> mkdir work ## the default work_dir
> cntk.exe configFile=atis.cntk
```

For Microsoft users only, to run the job on Philly:
- first upload data folder to philly cloud. e.g. `\\storage.gcr.philly.selfhost.corp.microsoft.com\pnrsy\<your_alias>\ATIS `
- update the config file to philly cloud, e.g. `\\storage.gcr.philly.selfhost.corp.microsoft.com\pnrsy_scratch\<your_alias>\ATIS`
- go to http://philly/ to create a new job by specifying data folder and config file, and start the job.

By default, the maxEpochs is set to 1. In order to get a good model accuracy, one can change it to larger value such as 20. 
Once the job starts, it should take about 20 minutes to run 20 epochs on single GPU, and slot F1 score is about 94.

More details about Philly, including how to upload data to Philly and start jobs, can be found [here](https://microsoft.sharepoint.com/teams/ATISG/SitePages/Philly%20Users%20Guide.aspx)



