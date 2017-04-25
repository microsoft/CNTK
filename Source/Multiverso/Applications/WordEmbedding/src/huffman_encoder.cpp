#include "huffman_encoder.h"

namespace wordembedding {

  HuffmanEncoder::HuffmanEncoder() {
    dict_ = nullptr;
  }

  //Save the word-huffmancode pair in file
  void HuffmanEncoder::Save2File(const char* filename) {
    FILE* fid = fopen(filename, "w");
    if (fid) {
      fprintf(fid, "%lld\n", hufflabel_info_.size());

      for (unsigned i = 0; i < hufflabel_info_.size(); ++i) {
        auto info = hufflabel_info_[i];
        const auto word = dict_->GetWordInfo(i);
        fprintf(fid, "%s %d", word->word.c_str(), info.codelen);

        for (int j = 0; j < info.codelen; ++j)
          fprintf(fid, " %d", info.code[j]);

        for (int j = 0; j < info.codelen; ++j)
          fprintf(fid, " %d", info.point[j]);

        fprintf(fid, "\n");
      }

      fclose(fid);
    }
    else {
      multiverso::Log::Error("file open failed %s", filename);
    }
  }

  //Recover the word-huffmancode pair from file
  void HuffmanEncoder::RecoverFromFile(const char* filename) {
    dict_ = new (std::nothrow)Dictionary();
    assert(dict_ != nullptr);
    FILE* fid;
    fid = fopen(filename, "r");
    if (fid) {
      int64 vocab_size;
      fscanf(fid, "%lld", &vocab_size);
      hufflabel_info_.reserve(vocab_size);
      hufflabel_info_.clear();

      int tmp;
      char sz_label[kMaxWordSize];
      for (int64 i = 0; i < vocab_size; ++i) {
        HuffLabelInfo info;

        //fscanf_s(fid, "%s", sz_label, kMaxWordSize);
        fscanf(fid, "%s", sz_label, kMaxWordSize);
        dict_->Insert(sz_label);

        fscanf(fid, "%d", &info.codelen);

        info.code.clear();
        info.point.clear();

        for (int j = 0; j < info.codelen; ++j) {
          fscanf(fid, "%d", &tmp);
          info.code.push_back(tmp);
        }
        for (int j = 0; j < info.codelen; ++j) {
          fscanf(fid, "%d", &tmp);
          info.point.push_back(tmp);
        }

        hufflabel_info_.push_back(info);
      }
      fclose(fid);
    }
    else {
      multiverso::Log::Error("file open failed %s", filename);
    }
  }
  //Compare the second element of two pairs   
  bool compare(const std::pair<int, int64>& x,
    const std::pair<int, int64>& y) {
    if (x.second == 0) return true;
    if (y.second == 0) return false;
    return (x.second > y.second);
  }
  //Build huffaman tree from the existing dictionary
  void HuffmanEncoder::BuildHuffmanTreeFromDict() {
    std::vector<std::pair<int, int64> > ordered_words;
    ordered_words.reserve(dict_->Size());
    ordered_words.clear();
    for (int i = 0; i < dict_->Size(); ++i)
      ordered_words.push_back(std::pair<int, int64>(i,
      dict_->GetWordInfo(i)->freq));
    std::sort(ordered_words.begin(), ordered_words.end(), compare);

    unsigned vocab_size = (unsigned)ordered_words.size();
    // frequence
    int64 *count = new (std::nothrow)int64[vocab_size * 2 + 1];
    assert(count != nullptr);
    // Huffman code relative to parent node [1,0] of each node
    unsigned *binary = new (std::nothrow)unsigned[vocab_size * 2 + 1];
    assert(binary != nullptr);
    memset(binary, 0, sizeof(unsigned)* (vocab_size * 2 + 1));

    unsigned *parent_node = new (std::nothrow)unsigned[vocab_size * 2 + 1];
    assert(parent_node != nullptr);
    memset(parent_node, 0, sizeof(unsigned)* (vocab_size * 2 + 1));
    unsigned code[kMaxCodeLength], point[kMaxCodeLength];

    for (unsigned i = 0; i < vocab_size; ++i)
      count[i] = ordered_words[i].second;
    for (unsigned i = vocab_size; i < vocab_size * 2; i++)
      count[i] = static_cast<int64>(1e15);
    int pos1 = vocab_size - 1;
    int pos2 = vocab_size;
    int min1i, min2i;
    for (unsigned i = 0; i < vocab_size - 1; i++) {
      // First, find two smallest nodes 'min1, min2'
      assert(pos2 < static_cast<int>(vocab_size)* 2 - 1);
      //Find the samllest node
      if (pos1 >= 0) {
        if (count[pos1] < count[pos2]) {
          min1i = pos1;
          pos1--;
        }
        else {
          min1i = pos2;
          pos2++;
        }
      }
      else {
        min1i = pos2;
        pos2++;
      }

      //Find the second samllest node
      if (pos1 >= 0) {
        if (count[pos1] < count[pos2]) {
          min2i = pos1;
          pos1--;
        }
        else {
          min2i = pos2;
          pos2++;
        }
      }
      else {
        min2i = pos2;
        pos2++;
      }

      count[vocab_size + i] = count[min1i] + count[min2i];

      assert(min1i >= 0);
      assert(min1i < static_cast<int>(vocab_size)* 2 - 1);
      assert(min2i >= 0);
      assert(min2i < static_cast<int>(vocab_size)* 2 - 1);
      parent_node[min1i] = vocab_size + i;
      parent_node[min2i] = vocab_size + i;
      binary[min2i] = 1;
    }
    assert(pos1 < 0);

    //Generate the huffman code for each leaf node
    hufflabel_info_.clear();
    for (unsigned a = 0; a < vocab_size; ++a)
      hufflabel_info_.push_back(HuffLabelInfo());
    for (unsigned a = 0; a < vocab_size; a++) {
      unsigned b = a, i = 0;
      while (1) {
        assert(i < kMaxCodeLength);
        code[i] = binary[b];
        point[i] = b;
        i++;
        b = parent_node[b];
        if (b == vocab_size * 2 - 2) break;
      }
      unsigned cur_word = ordered_words[a].first;

      hufflabel_info_[cur_word].codelen = i;
      hufflabel_info_[cur_word].point.push_back(vocab_size - 2);

      for (b = 0; b < i; b++) {
        hufflabel_info_[cur_word].code.push_back(code[i - b - 1]);
        if (b)
          hufflabel_info_[cur_word].point.push_back(point[i - b] - vocab_size);
      }
    }

    delete[] count;
    count = nullptr;
    delete[] binary;
    binary = nullptr;
    delete[] parent_node;
    parent_node = nullptr;
  }
  //Firstly get the dictionary from file
  void HuffmanEncoder::BuildFromTermFrequency(const char* filename) {
    FILE* fid;
    fid = fopen(filename, "r");
    if (fid) {
      char sz_label[kMaxWordSize];
      dict_ = new (std::nothrow)Dictionary();
      assert(dict_ != nullptr);
      //while (fscanf_s(fid, "%s", sz_label, kMaxWordSize) != EOF)
      while (fscanf(fid, "%s", sz_label) != EOF) {
        HuffLabelInfo info;
        int freq;
        fscanf(fid, "%d", &freq);
        dict_->Insert(sz_label, freq);
      }
      fclose(fid);

      BuildHuffmanTreeFromDict();
    }
    else {
      multiverso::Log::Error("file open failed %s", filename);
    }
  }

  void HuffmanEncoder::BuildFromTermFrequency(Dictionary* dict) {
    dict_ = dict;
    BuildHuffmanTreeFromDict();
  }

  int HuffmanEncoder::GetLabelSize() {
    return dict_->Size();
  }
  //Get the label index
  int HuffmanEncoder::GetLabelIdx(const char* label) {
    return dict_->GetWordIdx(label);
  }

  HuffLabelInfo* HuffmanEncoder::GetLabelInfo(char* label) {
    int idx = GetLabelIdx(label);
    if (idx == -1)
      return nullptr;
    return GetLabelInfo(idx);
  }

  HuffLabelInfo* HuffmanEncoder::GetLabelInfo(int label_idx) {
    if (label_idx == -1) return nullptr;
    return &hufflabel_info_[label_idx];
  }
  //Get the dictionary
  Dictionary* HuffmanEncoder::GetDict() {
    return dict_;
  }
}