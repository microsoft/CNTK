import io
import os
import re
import requests
import sys
import tarfile
import zipfile
import shutil
from io import open

try:
  from .wordvocab import *
except Exception:
  from wordvocab import *

def file_exists(src):
  return (os.path.isfile(src) and os.path.exists(src))

def download(src, target, cookies=None):
  target_dir=os.path.dirname(target)
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)    
  url=src
  print("Start to download data from {0} to {1}".format(url, target))
  response = requests.get(url, stream=True, cookies=cookies)
  with open(target, 'wb') as handle:
    for data in response.iter_content(chunk_size=2**20):
      if data: handle.write(data)
      sys.stdout.write('.')
      sys.stdout.flush()
  print()
  print("Finished to download {0} to {1}".format(url, target))

def merge_files(folder, target):
  if os.path.exists(target):
    return
  count = 0
  all_files = os.listdir(folder)
  print("Start to merge {0} files under folder {1} as {2}".format(len(all_files), folder, target))
  for f in all_files:
    txt=os.path.join(folder, f)
    if os.path.isfile(txt):
      with open(txt, 'r', encoding='utf-8') as sample:
        content = sample.readlines()
        context = content[2].strip()
        query = content[4].strip()
        answer = content[6].strip()
        entities = []
        for k in range(8, len(content)):
          entities += [ content[k].strip() ]
        with open(target, 'a', encoding='utf-8') as output:
          output.write(u"{0}\t{1}\t{2}\t{3}\n".format(query, answer, context, "\t".join(entities)))
    count+=1
    if count%1000==0:
      sys.stdout.write(".")
      sys.stdout.flush()
  print()
  print("Finished to merge {0}".format(target))


def download_cnn(target="."):
  if os.path.exists(os.path.join(target, "cnn")):
    shutil.rmtree(os.path.join(target, "cnn"))
  if not os.path.exists(target):
    os.makedirs(target)
  tar_target = os.path.join(target, "cnn.tar.gz")
  if not file_exists(tar_target):
    url="https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTTljRDVZMFJnVWM"
    print("Start to download CNN data from {0} to {1}".format(url, target))    
    pre_request = requests.get(url)
    confirm_match = re.search(r"confirm=(.{4})", pre_request.content.decode("utf-8"))
    confirm_url = url + "&confirm=" + confirm_match.group(1)
    download(confirm_url, tar_target, cookies=pre_request.cookies)  
  tar = tarfile.open(tar_target, mode="r:gz")
  tar.extractall(target)
  print("Finished to download {0} to {1}".format(url, target))

def download_glove_retrained_embedding(target="."):
  url="http://nlp.stanford.edu/data/glove.6B.zip"
  if os.path.exists(os.path.join(target, "glove")):
    shutil.rmtree(os.path.join(target, "glove"))
  zip_target = os.path.join(target, "glove.6B.zip") 
  target=os.path.join(target, "glove")
  if not os.path.exists(target):
    os.makedirs(target)
  print("Start to download GloVe pretrained embedding data from {0} to {1}".format(url, target))
  
  if not file_exists(zip_target):
    download(url, zip_target)
  zipf = zipfile.ZipFile(zip_target, mode='r')

  zipf.extractall(target)
  print("Finished to download {0} to {1}".format(url, target))

def prepare_data():
  py_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
  data_path = os.path.join(py_path, "Data")
  raw_train_data=os.path.join(data_path, "cnn/training.txt")
  raw_test_data=os.path.join(data_path, "cnn/test.txt")
  raw_validation_data=os.path.join(data_path, "cnn/validation.txt")
  if not (file_exists(raw_train_data) and file_exists(raw_test_data) and file_exists(raw_validation_data)):
    download_cnn(data_path)
  
  merge_files(os.path.join(data_path, "cnn/questions/training"), raw_train_data)
  merge_files(os.path.join(data_path, "cnn/questions/test"), raw_test_data)
  merge_files(os.path.join(data_path, "cnn/questions/validation"), raw_validation_data)
  print("All necessary cnn data are downloaded to {0}".format(data_path))
  
  vocab_path=os.path.join(data_path, "cnn/cnn.vocab")
  train_ctf=os.path.join(data_path, "cnn/training.ctf")
  test_ctf=os.path.join(data_path, "cnn/test.ctf")
  validation_ctf=os.path.join(data_path, "cnn/validation.ctf")
  vocab_size=101000
  if not (file_exists(train_ctf) and file_exists(test_ctf) and file_exists(validation_ctf)):
    entity_vocab, word_vocab = Vocabulary.build_vocab(raw_train_data, vocab_path, vocab_size)
    Vocabulary.build_corpus(entity_vocab, word_vocab, raw_train_data, train_ctf)
    Vocabulary.build_corpus(entity_vocab, word_vocab, raw_test_data, test_ctf)
    Vocabulary.build_corpus(entity_vocab, word_vocab, raw_validation_data, validation_ctf)
  print("Training data conversion finished.")
  
  if not file_exists(os.path.join(data_path, "glove/glove.6B.300d.txt")):
    download_glove_retrained_embedding(data_path)
  print("Glove embedding data downloaded.")
