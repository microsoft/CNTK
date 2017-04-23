#ifndef WORDEMBEDDING_DICTIONARY_H_
#define WORDEMBEDDING_DICTIONARY_H_
/*!
* \brief Class dictionary stores the vocabulary and it's frequency
*/

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstring>

#include <multiverso/util/log.h>

#include "constant.h"

namespace wordembedding {
  /*!
  * \brief struct WordInfo stores the pair of word&freq
  */
  struct WordInfo {
    std::string word;
    int64 freq;
    WordInfo() {
      freq = 0;
      word.clear();
    }
    WordInfo(const std::string& _word, int64 _freq) {
      word = _word;
      freq = _freq;
    }
  };

  class Dictionary {
  public:
    Dictionary();
    Dictionary(int i);
    void Clear();
    /*!
    * \brief Assign value to the set word_whitelist_
    */
    void SetWhiteList(const std::vector<std::string>& whitelist);
    /*!
    * \brief Remove the low-freq word
    */
    void RemoveWordsLessThan(int64 min_count);
    /*!
    * \brief Merge in the frequent words according to threshold
    */
    void MergeInfrequentWords(int64 threshold);
    /*!
    * \brief Insert word-freq pair to the dictionary
    * \param word the word string
    * \param cnt the word's frequency
    */
    void Insert(const char* word, int64 cnt = 1);
    /*!
    * \brief Load the word-freq pair from file
    */
    void LoadFromFile(const char* filename);
    void LoadTriLetterFromFile(const char* filename,
      unsigned int min_cnt = 1, unsigned int letter_count = 3);
    int GetWordIdx(const char* word);
    /*!
    * \brief Get the index of the word according to the dictionary
    */
    const WordInfo* GetWordInfo(const char* word);
    const WordInfo* GetWordInfo(int word_idx);
    int Size();
    void StartIteration();
    /*!
    * \brief Judge the word_iterator_ is the end
    */
    bool HasMore();
    /*!
    * \brief Get the next wordinfo pointer in the vector
    */
    const WordInfo* Next();
    std::vector<WordInfo>::iterator Begin();
    std::vector<WordInfo>::iterator End();

    void PrintVocab();

  private:
    int combine_;
    std::vector<WordInfo> word_info_;
    std::vector<WordInfo>::iterator word_iterator_;
    std::unordered_map<std::string, int> word_idx_map_;
    std::unordered_set<std::string> word_whitelist_;
  };
}
#endif