#include "wordembedding.h"

namespace wordembedding {

  WordEmbedding::WordEmbedding(Option* option, HuffmanEncoder* huffmanEncoder,
    Sampler* sampler, int dictionary_size) {
    word_count_actual = 0;
    option_ = option;
    huffmanEncoder_ = huffmanEncoder;
    sampler_ = sampler;
    dictionary_size_ = dictionary_size;
    learning_rate = option_->init_learning_rate;
    data_block_ = nullptr;
  }

  //Train neural networks of WordEmbedding
  void WordEmbedding::Train(DataBlock *data_block, int index_start, int interval,
    int64& word_count, real* hidden_act, real* hidden_err) {
    data_block_ = data_block;
    std::vector <int> negativesample(data_block_->negativesample_pools.begin(),
      data_block_->negativesample_pools.end());

    for (int i = index_start; i < data_block_->Size(); i += interval) {
      int sentence_length;
      int64 word_count_deta;
      int *sentence;
      uint64 next_random;
      data_block_->GetSentence(i, sentence, sentence_length,
        word_count_deta, next_random);

      this->Train(sentence, sentence_length,
        next_random, hidden_act, hidden_err, negativesample);

      word_count += word_count_deta;
    }
  }
  //Update the learning rate
  void WordEmbedding::UpdateLearningRate() {
    if (option_->use_adagrad == false) {
      learning_rate = static_cast<real>(option_->init_learning_rate *
        (1 - word_count_actual / (static_cast<double>(option_->total_words *
        option_->epoch) + 1.0)));
      if (learning_rate < option_->init_learning_rate * 0.0001)
        learning_rate = (static_cast<double>(option_->init_learning_rate)
        * 0.0001);
    }
  }

  void WordEmbedding::Train(int* sentence, int sentence_length,
    uint64 next_random, real* hidden_act, real* hidden_err,
    std::vector <int> &negativesample_pools) {
    ParseSentence(sentence, sentence_length,
      next_random, hidden_act, hidden_err,
      &WordEmbedding::TrainSample, negativesample_pools);
  }
  //Train with forward direction and get  the input-hidden layer vector
  void WordEmbedding::FeedForward(std::vector<int>& input_nodes,
    real* hidden_act) {
    for (int i = 0; i < input_nodes.size(); ++i) {
      int &node_id = input_nodes[i];
      real* input_embedding = GetWeightIE(node_id);
      for (int j = 0; j < option_->embeding_size; ++j) {
        hidden_act[j] += input_embedding[j];
      }
    }

    if (input_nodes.size() > 1) {
      for (int j = 0; j < option_->embeding_size; ++j) {
        hidden_act[j] /= input_nodes.size();
      }
    }
  }
  //Train with inverse direction and update the hidden-output 
  void WordEmbedding::BPOutputLayer(int label, int word_idx,
    real* classifier, real* hidden_act, real* hidden_err) {
    assert(classifier != nullptr && hidden_act != nullptr
      && hidden_err != nullptr);
    real f = 0.0f;

    //Propagate hidden -> output
    for (int j = 0; j < option_->embeding_size; ++j)
      f += hidden_act[j] * classifier[j];

    f = 1 / (1 + exp(-f));

    real error = 0.0f;
    if (option_->hs) {
      error = (1 - label - f);
    }

    if (option_->negative_num) {
      error = label - f;
    }

    //Propagate errors output -> hidden
    for (int j = 0; j < option_->embeding_size; ++j)
      hidden_err[j] += error * classifier[j];

    if (option_->use_adagrad) {
      real* sum_gradient2_row = GetSumGradient2EO(word_idx);
      assert(sum_gradient2_row != nullptr);
      //Learn weights hidden -> output
      for (int j = 0; j < option_->embeding_size; ++j) {
        real g = error * hidden_act[j];
        sum_gradient2_row[j] += g * g;
        if (sum_gradient2_row[j] > 1e-10)
          classifier[j] += g * option_->init_learning_rate /
          sqrt(sum_gradient2_row[j]);
      }
    }
    else {
      //'g' is the gradient multiplied by the learning rate
      real g = error * learning_rate;
      //Learn weights hidden -> output
      for (int j = 0; j < option_->embeding_size; ++j)
        classifier[j] += g * hidden_act[j];
    }
  }

  void WordEmbedding::TrainSample(std::vector<int>& input_nodes,
    std::vector<std::pair<int, int> >& output_nodes,
    void *local_hidden_act, void *local_hidden_err) {
    real* hidden_act = (real*)local_hidden_act;
    real* hidden_err = (real*)local_hidden_err;
    assert(hidden_act != nullptr);
    assert(hidden_err != nullptr);
    memset(hidden_act, 0, option_->embeding_size * sizeof(real));
    memset(hidden_err, 0, option_->embeding_size * sizeof(real));

    FeedForward(input_nodes, hidden_act);

    for (int i = 0; i < output_nodes.size(); ++i) {
      int &node_id = output_nodes[i].first;
      int &code = output_nodes[i].second;
      BPOutputLayer(code, node_id, GetWeightEO(node_id),
        hidden_act, hidden_err);
    }

    if (option_->use_adagrad) {
      //Update context embedding
      for (int i = 0; i < input_nodes.size(); ++i) {
        int &node_id = input_nodes[i];
        real* input_embedding_row = GetWeightIE(node_id);
        real* sum_gradient2_row = GetSumGradient2IE(node_id);
        assert(input_embedding_row != nullptr && sum_gradient2_row != nullptr);
        for (int j = 0; j < option_->embeding_size; ++j) {
          sum_gradient2_row[j] += hidden_err[j] * hidden_err[j];
          if (sum_gradient2_row[j] > 1e-10)
            input_embedding_row[j] += hidden_err[j] *
            option_->init_learning_rate / sqrt(sum_gradient2_row[j]);
        }
      }
    }
    else {
      for (int j = 0; j < option_->embeding_size; ++j)
        hidden_err[j] *= learning_rate;
      //Update context embedding
      for (int i = 0; i < input_nodes.size(); ++i) {
        int &node_id = input_nodes[i];
        real* input_embedding = GetWeightIE(node_id);
        assert(input_embedding != nullptr);
        for (int j = 0; j < option_->embeding_size; ++j)
          input_embedding[j] += hidden_err[j];
      }
    }
  }

  //Parapare the data for the datablock
  void WordEmbedding::PrepareData(DataBlock* data_block) {
    int sentence_length;
    int64 word_count_delta;
    int *sentence;
    uint64 next_random;
    if (option_->hs) {
      for (int i = 0; i < data_block->Size(); ++i) {
        data_block->GetSentence(i, sentence, sentence_length,
          word_count_delta, next_random);

        for (int sentence_position = 0; sentence_position < sentence_length;
          ++sentence_position) {
          data_block->input_nodes.insert(sentence[sentence_position]);
        }
      }
      for (auto input_node : data_block->input_nodes) {
        auto info = huffmanEncoder_->GetLabelInfo(input_node);
        for (int d = 0; d < info->codelen; d++)
          data_block->output_nodes.insert(info->point[d]);
      }
    }
    else {
      for (int i = 0; i < data_block->Size(); ++i) {
        data_block->GetSentence(i, sentence, sentence_length,
          word_count_delta, next_random);

        for (int sentence_position = 0; sentence_position < sentence_length;
          ++sentence_position) {
          data_block->input_nodes.insert(sentence[sentence_position]);
        }
      }
      for (auto input_node : data_block->input_nodes) {
        data_block->output_nodes.insert(input_node);
      }
      for (int d = 0; d < option_->negative_num * data_block->input_nodes.size();
        d++) {
        next_random = sampler_->GetNextRandom(next_random);
        int target = sampler_->NegativeSampling(next_random);
        data_block->output_nodes.insert(target);
        data_block->negativesample_pools.insert(target);
      }
    }

  multiverso::Log::Debug("PrepareData input_nodes size is %d , output_nodes size is %d", data_block->input_nodes.size(), data_block->output_nodes.size());
  }

  //Parse the sentence and deepen into two branches
  void WordEmbedding::ParseSentence(int* sentence, int sentence_length,
    uint64 next_random, real* hidden_act, real* hidden_err,
    FunctionType function, std::vector <int> &negativesample_pools) {
    if (sentence_length == 0)
      return;

    int feat[kMaxSentenceLength + 1];
    std::vector<int> input_nodes;
    std::vector<std::pair<int, int> > output_nodes;
    for (int sentence_position = 0; sentence_position < sentence_length;
      ++sentence_position) {
      if (sentence[sentence_position] == -1) continue;
      next_random = sampler_->GetNextRandom(next_random);
      int off = next_random % option_->window_size;
      int feat_size = 0;
      for (int i = off; i < option_->window_size * 2 + 1 - off; ++i)
      if (i != option_->window_size) {
        int c = sentence_position - option_->window_size + i;
        if (c < 0 || c >= sentence_length || sentence[c] == -1)
          continue;

        feat[feat_size++] = sentence[c];
        //train Skip-gram
        if (!option_->cbow) {
          input_nodes.clear();
          output_nodes.clear();
          Parse(feat + feat_size - 1, 1, sentence[sentence_position],
            next_random, input_nodes, output_nodes, negativesample_pools);
          (this->*function)(input_nodes, output_nodes, hidden_act, hidden_err);
        }
      }

      //train cbow
      if (option_->cbow) {
        input_nodes.clear();
        output_nodes.clear();
        Parse(feat, feat_size, sentence[sentence_position],
          next_random, input_nodes, output_nodes, negativesample_pools);
        (this->*function)(input_nodes, output_nodes, hidden_act, hidden_err);
      }
    }
  }
  //Parse the windows's input&output nodes
  inline void WordEmbedding::Parse(int *feat, int feat_cnt, int word_idx,
    uint64 &next_random, std::vector<int>& input_nodes,
    std::vector<std::pair<int, int> >& output_nodes,
    std::vector <int> &negativesample_pools) {
    for (int i = 0; i < feat_cnt; ++i) {
      input_nodes.push_back(feat[i]);
    }

    if (option_->hs) {
      auto info = huffmanEncoder_->GetLabelInfo(word_idx);
      for (int d = 0; d < info->codelen; d++)
        output_nodes.push_back(std::make_pair(info->point[d], info->code[d]));
    }
    else
    if (option_->negative_num) {
      output_nodes.push_back(std::make_pair(word_idx, 1));
      for (int d = 0; d < option_->negative_num; d++) {
        next_random = sampler_->GetNextRandom(next_random);
        int index = (next_random >> 8) % negativesample_pools.size();
        int target = negativesample_pools[index];
        if (target == word_idx) continue;
        output_nodes.push_back(std::make_pair(target, 0));
      }
    }
  }

  void WordEmbedding::SetWeightIE(int input_node_id, real* ptr) {
    data_block_->SetWeightIE(input_node_id, ptr);
  }

  void WordEmbedding::SetWeightEO(int output_node_id, real* ptr) {
    data_block_->SetWeightEO(output_node_id, ptr);
  }

  real* WordEmbedding::GetWeightIE(int input_node_id) {
    return data_block_->GetWeightIE(input_node_id);
  }

  real* WordEmbedding::GetWeightEO(int output_node_id) {
    return data_block_->GetWeightEO(output_node_id);
  }

  void WordEmbedding::SetSumGradient2IE(int input_node_id, real* ptr) {
    data_block_->SetSumGradient2IE(input_node_id, ptr);
  }

  void WordEmbedding::SetSumGradient2EO(int output_node_id, real* ptr) {
    data_block_->SetSumGradient2EO(output_node_id, ptr);
  }

  real* WordEmbedding::GetSumGradient2IE(int input_node_id) {
    return data_block_->GetSumGradient2IE(input_node_id);
  }

  real* WordEmbedding::GetSumGradient2EO(int output_node_id) {
    return data_block_->GetSumGradient2EO(output_node_id);
  }
}