#ifndef WORDEMBEDDING_DATA_BLOCK_H_
#define WORDEMBEDDING_DATA_BLOCK_H_

/*!
* \file data_block.h
* \brief Class DataBlock is to store the necessary data for trainer and param_loader
*/

#include<iostream>

#include "util.h"
#include "huffman_encoder.h"
#include "constant.h"

namespace wordembedding {
  /*!
  * \brief The class DataBlock stores the input nodes ,output nodes
  and their parameters.
  */
  class DataBlock {
  public:
    std::unordered_set <int> input_nodes, output_nodes;
    std::unordered_set <int> negativesample_pools;

    DataBlock(){}
    ~DataBlock();

    /*!
    * \brief Get the number of sentences stored in DataBlock
    * \return the number of sentences
    */
    size_t Size();
    /*!
    * \brief Add a new sentence to the DataBlock
    * \param sentence the starting address of the sentence
    * \param sentence_length the length of the sentence
    * \param word_count the number of words when getting the
    *        sentence from train-file
    * \param next_random the seed for getting random number
    */
    void AddSentence(int *sentence, int sentence_length,
      int64 word_count, uint64 next_random);
    /*!
    * \brief Get the information of the index-th sentence
    * \param index the id of the sentence
    * \param sentence the starting address of the sentence
    * \param sentence_length the length of the sentence
    * \param word_count the number of words when getting the
    *        sentence from train-file
    * \param next_random the seed for getting random number
    */
    void GetSentence(int index, int* &sentence,
      int &sentence_length, int64 &word_count,
      uint64 &next_random);

    /*!
    * \brief Release the memory which are using to store sentences
    */
    void ClearSentences();
    /*!
    * \brief Release the memory which are using to parameters
    */
    void ClearParameters();

    void MallocMemory(int dictionary_size, bool is_use_adagrad);

    void  SetWeightIE(int input_node_id, real* ptr);
    void  SetWeightEO(int output_node_id, real* ptr);
    real* GetWeightIE(int input_node_id);
    real* GetWeightEO(int output_node_id);

    void SetSumGradient2IE(int input_node_id, real* ptr);
    void SetSumGradient2EO(int output_node_id, real* ptr);
    real* GetSumGradient2IE(int input_node_id);
    real* GetSumGradient2EO(int output_node_id);

    void SetLastFlag();
    bool isLast();

  private:
    /*!
    * \brief The information of sentences
    * head the head address which store the sentence
    * length the number of words in the sentence
    * word_count the real word count of the sentence
    * next_random the random seed
    */
    struct Sentence {
      int* head;
      int length;
      int64 word_count;
      uint64 next_random;
      Sentence(int *head, int length, int64 word_count,
        uint64 next_random) :head(head), length(length),
        word_count(word_count), next_random(next_random){}

      void Get(int* &local_head, int &sentence_length,
        int64 &local_word_count, uint64 &local_next_random) {
        local_head = head;
        sentence_length = length;
        local_word_count = word_count;
        local_next_random = next_random;
      }
    };

    /*! \brief Store the information of sentences*/
    std::vector <Sentence> sentences_;

    real** weight_IE_ = nullptr;
    real** weight_EO_ = nullptr;

    real** sum_gradient2_IE_ = nullptr;
    real** sum_gradient2_EO_ = nullptr;

    bool is_use_adagrad_ = false;
    bool is_last_one_ = false;

    // No copying allowed
    DataBlock(const DataBlock&);
  };
}
#endif