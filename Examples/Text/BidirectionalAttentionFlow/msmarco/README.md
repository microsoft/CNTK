# Instructions for getting started with bidaf

## Download the MS MARCO  dataset
```
wget https://msmarco.blob.core.windows.net/msmarco/train_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v1.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/test_public_v1.1.json.gz
```
## Download GloVe vectors
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
## Download NLTK punkt
```
python3 -m nltk.downloader -d $HOME/nltk_data punkt
```
or
```
python -m nltk.downloader -d %USERPROFILE%/nltk_data punkt
```

## Run `convert_msmarco.py`
It should create `train.tsv`, `dev.tsv`, and `test.tsv` and `vocab.pkl`

It does a fair amount of preprocessing therefore converting data to cntk text format reader starts from these files

## Run `tsv2ctf.py`
It creates a `vocabs.pkl` and `train.ctf`, `val.ctf`, and `dev.ctf`

The data is ready now. Run train_pm.py to create the cntk model.
