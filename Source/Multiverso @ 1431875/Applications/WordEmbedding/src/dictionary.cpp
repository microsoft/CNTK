#include "dictionary.h"

namespace wordembedding {

  Dictionary::Dictionary() {
    combine_ = 0;
    Clear();
  }

  Dictionary::Dictionary(int i) {
    combine_ = i;
    Clear();
  }

  void Dictionary::Clear() {
    word_idx_map_.clear();
    word_info_.clear();
    word_whitelist_.clear();
  }
  //Set the white list for the dictionary
  void Dictionary::SetWhiteList(const std::vector<std::string>& whitelist) {
    for (unsigned int i = 0; i < whitelist.size(); ++i)
      word_whitelist_.insert(whitelist[i]);
  }
  //Merge in the word_info which has the frequency over-threshold
  void Dictionary::MergeInfrequentWords(int64 threshold) {
    word_idx_map_.clear();
    std::vector<WordInfo> tmp_info;
    tmp_info.clear();
    int infreq_idx = -1;

    for (auto word_info : word_info_) {
      if (word_info.freq >= threshold || word_info.freq == 0
        || word_whitelist_.count(word_info.word)) {
        word_idx_map_[word_info.word] = static_cast<int>(tmp_info.size());
        tmp_info.push_back(word_info);
      }
      else {
        if (infreq_idx < 0) {
          WordInfo infreq_word_info;
          infreq_word_info.word = "WE_ARE_THE_INFREQUENT_WORDS";
          infreq_word_info.freq = 0;
          word_idx_map_[infreq_word_info.word] = static_cast<int>(tmp_info.size());
          infreq_idx = static_cast<int>(tmp_info.size());
          tmp_info.push_back(infreq_word_info);
        }
        word_idx_map_[word_info.word] = infreq_idx;
        tmp_info[infreq_idx].freq += word_info.freq;
      }
    }
    word_info_ = tmp_info;
  }
  //Remove the words with frequency under min_count
  void Dictionary::RemoveWordsLessThan(int64 min_count) {
    word_idx_map_.clear();
    std::vector<WordInfo> tmp_info;
    tmp_info.clear();
    for (auto info : word_info_) {
      if (info.freq >= min_count || info.freq == 0
        || word_whitelist_.count(info.word)) {
        word_idx_map_[info.word] = static_cast<int>(tmp_info.size());
        tmp_info.push_back(info);
      }
    }
    word_info_ = tmp_info;
  }
  //Insert the dictionary element
  void Dictionary::Insert(const char* word, int64 cnt) {
    auto it = word_idx_map_.find(word);
    if (it != word_idx_map_.end())
      word_info_[it->second].freq += cnt;
    else {
      word_idx_map_[word] = static_cast<int>(word_info_.size());
      word_info_.push_back(WordInfo(word, cnt));
    }
  }
  //Load dictionary from file
  void Dictionary::LoadFromFile(const char* filename) {
    FILE* fid;
    fid = fopen(filename, "r");

    if (fid) {
      char sz_label[kMaxWordSize];

      //while ((fid, "%s", sz_label, kMaxWordSize) != EOF)
      while (fscanf(fid, "%s", sz_label, kMaxWordSize) != EOF) {
        int freq;
        fscanf(fid, "%d", &freq);
        Insert(sz_label, freq);
      }
      fclose(fid);
    }
  }

  void Dictionary::LoadTriLetterFromFile(const char* filename,
    unsigned int min_cnt, unsigned int letter_count) {
    FILE* fid;
    fid = fopen(filename, "r");
    if (fid) {
      char sz_label[kMaxWordSize] = { 0 };
      //while (fscanf_s(fid, "%s", sz_label, kMaxWordSize) != EOF)
      while (fscanf(fid, "%s", sz_label, kMaxWordSize) != EOF) {
        int64 freq;
        fscanf(fid, "%lld", &freq);
        if (freq < static_cast<int64>(min_cnt)) continue;

        // Construct Tri-letter From word
        size_t len = strlen(sz_label);
        if (len > kMaxWordSize) {
          multiverso::Log::Info("ignore super long term");
          continue;
        }

        char tri_letters[kMaxWordSize + 2];
        tri_letters[0] = '#';
        int i = 0;
        for (i = 0; i < strlen(sz_label); i++) {
          tri_letters[i + 1] = sz_label[i];
        }

        tri_letters[i + 1] = '#';
        tri_letters[i + 2] = 0;
        if (combine_) Insert(sz_label, freq);

        if (strlen(tri_letters) <= letter_count) {
          Insert(tri_letters, freq);
        }
        else {
          for (i = 0; i <= strlen(tri_letters) - letter_count; ++i) {
            char tri_word[kMaxWordSize];
            unsigned int j = 0;
            for (j = 0; j < letter_count; j++) {
              tri_word[j] = tri_letters[i + j];
            }
            tri_word[j] = 0;
            Insert(tri_word, freq);
          }
        }
      }
      fclose(fid);
    }
  }

  //Get the word's index from dictionary
  int Dictionary::GetWordIdx(const char* word) {
    auto it = word_idx_map_.find(word);
    if (it != word_idx_map_.end())
      return it->second;
    return -1;
  }
  //Return the size of frequency
  int Dictionary::Size() {
    return static_cast<int>(word_info_.size());
  }
  //Get the wordinfo from word or index
  const WordInfo* Dictionary::GetWordInfo(const char* word) {
    auto it = word_idx_map_.find(word);
    if (it != word_idx_map_.end())
      return GetWordInfo(it->second);
    return NULL;
  }

  const WordInfo* Dictionary::GetWordInfo(int word_idx) {
    if (word_idx >= 0 && word_idx < word_info_.size())
      return &word_info_[word_idx];
    return NULL;
  }

  void Dictionary::StartIteration() {
    word_iterator_ = word_info_.begin();
  }
  //Judge whether the iterator is the end
  bool Dictionary::HasMore() {
    return word_iterator_ != word_info_.end();
  }
  //Get the next Wordinfo
  const WordInfo* Dictionary::Next() {
    const WordInfo* entry = &(*word_iterator_);
    ++word_iterator_;
    return entry;
  }

  std::vector<WordInfo>::iterator Dictionary::Begin() {
    return word_info_.begin();
  }

  std::vector<WordInfo>::iterator Dictionary::End() {
    return word_info_.end();
  }
}
