#ifndef WORDEMBEDDING_HUFFMAN_ENCODER_H_
#define WORDEMBEDDING_HUFFMAN_ENCODER_H_
/*!
* \brief Class Huffman_encoder stores the huffman_encode of the vocabulary according the dictionary
*/

#include <algorithm>
#include <cassert>
#include <cstring>

#include "dictionary.h"
#include "constant.h"

namespace wordembedding {
  struct HuffLabelInfo
  {   /*!
    * \brief Internal node ids in the code path
    */
    std::vector<int> point;
    /*!
    * \brief Huffman code
    */
    std::vector<char> code;
    int codelen;
    HuffLabelInfo() {
      codelen = 0;
      point.clear();
      code.clear();
    }
  };

  class HuffmanEncoder {
  public:
    HuffmanEncoder();
    /*!
    * \brief Save the word-huffmancode in the file
    */
    void Save2File(const char* filename);
    /*!
    * \brief Recover the word-huffmancode from the file
    */
    void RecoverFromFile(const char* filename);
    /*!
    * \brief Get the dictionary file and build
    * \hufflabel_info from the dictionary
    */
    void BuildFromTermFrequency(const char* filename);
    void BuildFromTermFrequency(Dictionary* dict);
    /*!
    * \brief Get the label size
    */
    int GetLabelSize();
    /*!
    * \brief Get the label's index
    */
    int GetLabelIdx(const char* label);
    HuffLabelInfo* GetLabelInfo(char* label);
    HuffLabelInfo* GetLabelInfo(int label_idx);
    Dictionary* GetDict();

  private:
    void BuildHuffmanTreeFromDict();
    std::vector<HuffLabelInfo> hufflabel_info_;
    Dictionary* dict_;
  };
}
#endif