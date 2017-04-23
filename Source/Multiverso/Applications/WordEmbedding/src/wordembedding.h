#ifndef WORDEMBEDDING_WORD_EMBEDDING_H_
#define WORDEMBEDDING_WORD_EMBEDDING_H_
/*!
* file WordEmbedding.h
* \brief Class WordEmbedding includes some functions and parameters about TrainNN
*/

#include <vector>

#include "util.h"
#include "huffman_encoder.h"
#include "constant.h"
#include "data_block.h"

namespace wordembedding {

  class WordEmbedding {
  public:
    real learning_rate;
    int64 word_count_actual;

    WordEmbedding(Option* option, HuffmanEncoder* huffmanEncoder,
      Sampler* sampler, int dictionary_size);
    /*!
    * \brief TrainNN
    * \param data_block represents the trainNNing datablock
    * \param index_start the thread's starting index in the sentence vector
    * \param interval the total_number of thread
    * \param word_count count the words which has been processed by trainNN
    * \param hidden_act  hidden layer value
    * \param hidden_err  hidden layer error
    */
    void Train(DataBlock *data_block, int index_start,
      int interval, int64& word_count,
      real* hidden_act, real* hidden_err);
    /*!
    * \brief PrepareParameter for parameterloader threat
    * \param data_block datablock for parameterloader to parse
    */
    void PrepareData(DataBlock *data_block);
    /*!
    * \brief Update the learning rate
    */
    void UpdateLearningRate();
    /*!
    * \brief Set the input(output)-embedding weight
    */
    void SetWeightIE(int input_node_id, real* ptr);
    void SetWeightEO(int output_node_id, real* ptr);

    /*!
    * \brief Return the parametertable value
    */
    real* GetWeightIE(int input_node_id);
    real* GetWeightEO(int output_node_id);
    /*!
    * \brief Set the input(output) gradient-embedding weight when using adagrad
    */
    void SetSumGradient2IE(int input_node_id, real* ptr);
    void SetSumGradient2EO(int output_node_id, real* ptr);
    /*!
    * \brief Return the input(output) gradient-embedding weight when using adagrad
    */
    real* GetSumGradient2IE(int input_node_id);
    real* GetSumGradient2EO(int output_node_id);

  private:
    Option *option_ = nullptr;
    Dictionary *dictionary_ = nullptr;
    HuffmanEncoder *huffmanEncoder_ = nullptr;
    Sampler *sampler_ = nullptr;
    std::unordered_set<int> input_nodes_, output_nodes_;
    int dictionary_size_;

    DataBlock * data_block_ = nullptr;

    typedef void(WordEmbedding::*FunctionType)(std::vector<int>& input_nodes,
      std::vector<std::pair<int, int> >& output_nodes,
      void *hidden_act, void *hidden_err);
    /*!
    * \brief Parse the needed parameter in a window
    */
    void Parse(int *feat, int feat_cnt, int word_idx, uint64 &next_random,
      std::vector<int>& input_nodes,
      std::vector<std::pair<int, int> >& output_nodes, std::vector <int> &negativesample_pools);
    /*!
    * \brief Parse a sentence and deepen into two branches
    * \one for TrainNN,the other one is for Parameter_parse&request
    */
    void ParseSentence(int* sentence, int sentence_length,
      uint64 next_random,
      real* hidden_act, real* hidden_err,
      FunctionType function, std::vector <int> &negativesample_pools);
    /*!
    * \brief Get the hidden layer vector
    * \param input_nodes represent the input nodes
    * \param hidden_act store the hidden layer vector
    */
    void FeedForward(std::vector<int>& input_nodes, real* hidden_act);
    /*!
    * \brief Calculate the hidden_err and update the output-embedding weight
    * \param label record the label of every output-embedding vector
    * \param word_idx the index of the output-embedding vector
    * \param classifier store the output-embedding vector
    * \param hidden_act store the hidden layer vector
    * \param hidden_err store the hidden-error which is used
    * \to update the input-embedding vector
    */
    void BPOutputLayer(int label, int word_idx, real* classifier,
      real* hidden_act, real* hidden_err);

    /*!
    * \brief Train a window sample and update the
    * \input-embedding & output-embedding vectors
    * \param input_nodes represent the input nodes
    * \param output_nodes represent the output nodes
    * \param hidden_act  store the hidden layer vector
    * \param hidden_err  store the hidden layer error
    */
    void TrainSample(std::vector<int>& input_nodes,
      std::vector<std::pair<int, int> >& output_nodes,
      void *hidden_act, void *hidden_err);
    /*!
    * \brief Train the sentence actually
    */
    void Train(int* sentence, int sentence_length,
      uint64 next_random, real* hidden_act, real* hidden_err, std::vector <int> &negativesample_pools);

    //No copying allowed
    WordEmbedding(const WordEmbedding&);
    void operator=(const WordEmbedding&);
  };
}
#endif