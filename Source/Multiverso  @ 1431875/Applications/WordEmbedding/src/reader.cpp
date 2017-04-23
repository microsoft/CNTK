#include "reader.h"

namespace wordembedding {

  Reader::Reader(Dictionary *dictionary, Option *option,
    Sampler *sampler, const char *input_file) {
    dictionary_ = dictionary;
    option_ = option;
    sampler_ = sampler;

    stopwords_table_.clear();
    if (option_->stopwords) {
      FILE* fid = fopen(option_->sw_file, "r");
      if (fid == nullptr) {
        multiverso::Log::Fatal("Open sw_file failed!\n");
        exit(1);
      }
      while (ReadWord(word_, fid)) {
        stopwords_table_.insert(word_);
      }

      fclose(fid);
    }

    file_ = fopen(input_file, "r");
    if (file_ == nullptr) {
      multiverso::Log::Fatal("Open train_file failed!\n");
      exit(1);
    }
  }

  Reader::~Reader() {
    if (file_ != nullptr)
      fclose(file_);
  }
  //Get sentence by connecting the words extracted
  int Reader::GetSentence(int *sentence, int64 &word_count) {
    int length = 0, word_idx;
    word_count = 0;
    while (1) {
      if (!ReadWord(word_, file_))
        break;
      word_idx = dictionary_->GetWordIdx(word_);
      if (word_idx == -1)
        continue;
      word_count++;
      if (option_->stopwords && stopwords_table_.count(word_))
        continue;
      if (option_->sample > 0 &&
        !sampler_->WordSampling(
        dictionary_->GetWordInfo(word_idx)->freq,
        option_->total_words, option_->sample))
        continue;
      sentence[length++] = word_idx;
      if (length >= kMaxSentenceLength)
        break;
    }

    return length;
  }

  void Reader::ResetStart() {
    fseek(file_, 0, SEEK_SET);
  }

  void Reader::ResetSize(int64 size) {
    byte_count_ = 0;
    byte_size_ = size;
  }
  //Read words from the file
  bool Reader::ReadWord(char *word, FILE *fin) {
    int idx = 0;
    char ch;
    while (!feof(fin) && byte_count_ < byte_size_) {
      ch = fgetc(fin);
      ++byte_count_;
      if (ch == 13) continue;
      if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
        if (idx > 0) {
          if (ch == '\n')
            ungetc(ch, fin);
          break;
        }
        if (ch == '\n') {
          strcpy(word, (char *)"</s>");
          return true;
        }
        else continue;
      }
      word[idx++] = ch;
      //Truncate too long words
      if (idx >= kMaxString - 1)
        idx--;
    }
    word[idx] = 0;
    return idx != 0;
  }
}