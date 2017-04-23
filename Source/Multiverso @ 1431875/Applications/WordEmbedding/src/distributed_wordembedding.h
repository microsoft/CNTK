#ifndef WORDEMBEDDING_DISTRIBUTED_WORDEMBEDDING_H_
#define WORDEMBEDDING_DISTRIBUTED_WORDEMBEDDING_H_
/*!
* file distributed_wordembedding.h
* \brief Class Distributed_wordembedding describes the main frame of 
* WordEmbedding and some useful functions
*/

#include <vector>

#include <thread>
#include <functional>
#include <omp.h>

#include <multiverso/multiverso.h>

#include "util.h"
#include "huffman_encoder.h"
#include "reader.h"
#include "trainer.h"
#include "block_queue.h"
#include "communicator.h"
#include "wordembedding.h"

namespace wordembedding {

  extern std::string g_log_suffix;

  class DistributedWordembedding {
  public:
    /*!
    * \brief Run Function contains everything
    */
    void Run(int argc, char *argv[]);

  private:
    clock_t start_;
    int process_id_;
    Option* option_ = nullptr;
    Dictionary* dictionary_ = nullptr;
    HuffmanEncoder* huffman_encoder_ = nullptr;
    Sampler* sampler_ = nullptr;
    Reader* reader_ = nullptr;
    WordEmbedding* WordEmbedding_ = nullptr;
    BlockQueue *block_queue_ = nullptr;
    std::thread load_data_thread_;
    std::thread collect_wordcount_thread_;
    bool is_running_ = false;
    std::vector<Trainer*> trainers_;
    Communicator* communicator_ = nullptr;
    MemoryManager* memory_mamanger_ = nullptr;

    /*!
    * \brief Load Dictionary from the vocabulary_file
    * \param opt Some model-set setparams
    * \param dictionary save the vocabulary and its frequency
    * \param huffman_encoder convert dictionary to the huffman_code
    */
    int64 LoadVocab(Option *opt, Dictionary *dictionary,
      HuffmanEncoder *huffman_encoder);

    void Train(int argc, char *argv[]);
    void TrainNeuralNetwork();

    void PrepareData(DataBlock *data_block);

    void StartLoadDataThread(Reader *reader, int64 file_size);
    void LoadOneBlock(DataBlock *data_block,
      Reader *reader, int64 size);

    void StartCollectWordcountThread();
    void StopCollectWordcountThread();

    void StartWordCount();
    void GetAllWordCount();
    void AddDeltaWordCount();

    DataBlock* GetDataFromQueue();
    DataBlock* GetBlockAndPrepareParameter();

    void SaveEmbedding(const char *file_path, bool is_binary);
    void WriteToFile(bool is_binary, std::vector<real*> &blocks, FILE* fid,
      std::vector<int> &nodes);
    const char* ChangeFileName(const char *file_path, int iteration);
  };
}
#endif