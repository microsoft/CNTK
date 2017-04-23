#include "trainer.h"
namespace wordembedding {

  Trainer::Trainer(int trainer_id, Option *option,
    Dictionary* dictionary, WordEmbedding* WordEmbedding) {
    trainer_id_ = trainer_id;
    option_ = option;
    word_count = 0;
    WordEmbedding_ = WordEmbedding;
    dictionary_ = dictionary;
    hidden_act_ = (real *)calloc(option_->embeding_size, sizeof(real));
    hidden_err_ = (real *)calloc(option_->embeding_size, sizeof(real));
    process_count_ = -1;
    process_id_ = -1;

    assert(hidden_act_ != nullptr);
    assert(hidden_err_ != nullptr);
    start_ = 0;
    train_count_ = 0;
  }

  Trainer::~Trainer() {
    free(hidden_act_);
    free(hidden_err_);
  }

  void Trainer::TrainIteration(DataBlock *data_block) {
    if (process_id_ == -1)
      process_id_ = multiverso::MV_Rank();

    if (data_block == nullptr) {
      return;
    }

    int64 last_word_count = word_count;
    clock_t start = clock();

    multiverso::Log::Debug("Rank %d Train %d TrainNN Begin TrainIteration%d ...\n",
      process_id_, trainer_id_, train_count_);

    WordEmbedding_->Train(data_block, trainer_id_, option_->thread_cnt,
      word_count, hidden_act_, hidden_err_);

    if (word_count > last_word_count) {
      multiverso::Log::Info("Rank %d TrainNNSpeed: Words/thread/second %lfk\n",
        process_id_,
        (static_cast<double>(word_count)-last_word_count) /
        (clock() - start) * static_cast<double>(CLOCKS_PER_SEC) / 1000);
    }

    multiverso::Log::Debug("Rank %d Trainer %d training time:%lfs\n", process_id_,
      trainer_id_, (clock() - start) / static_cast<double>(CLOCKS_PER_SEC));
    train_count_++;
  }
}