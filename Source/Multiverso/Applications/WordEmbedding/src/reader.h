#ifndef WORDEMBEDDING_READER_H_
#define WORDEMBEDDING_READER_H_
/*!
* file reader.h
* \brief Class Reader helps the function Loaddata to fill the datablock
*/

#include <unordered_set>

#include "util.h"
#include "dictionary.h"
#include "constant.h"

namespace wordembedding {

  class Reader {
  public:
    Reader(Dictionary *dictionary, Option *option,
      Sampler *sampler, const char *input_file);
    ~Reader();
    /*!
    * \brief Getsentence from the train_file
    * \param sentence save the sentence by the word index according to the dictionary
    * \param word_count count the sentence length
    */
    int GetSentence(int *sentence, int64 &word_count);
    void ResetStart();
    void ResetSize(int64 size);

  private:
    const Option *option_;
    FILE* file_;
    char word_[kMaxString + 1];
    Dictionary *dictionary_;
    Sampler *sampler_;
    int64 byte_count_, byte_size_;
    std::unordered_set<std::string> stopwords_table_;
    /*!
    * \brief Read words from the train_file
    * \param word store the extracted word
    * \param file represent the train_file pointer
    */
    bool ReadWord(char *word, FILE *file);

    //No copying allowed
    Reader(const Reader&);
    void operator=(const Reader&);
  };
}
#endif