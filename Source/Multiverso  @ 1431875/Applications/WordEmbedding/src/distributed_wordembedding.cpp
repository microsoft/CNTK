#include "distributed_wordembedding.h"

namespace wordembedding {

  void DistributedWordembedding::LoadOneBlock(DataBlock *data_block,
    Reader *reader, int64 size) {
    clock_t start = clock();

    data_block->ClearSentences();
    reader->ResetSize(size);
    while (true) {
      int64 word_count = 0;
      int *sentence = new (std::nothrow)int[kMaxSentenceLength + 2];
      assert(sentence != nullptr);
      int sentence_length = reader->GetSentence(sentence, word_count);
      if (sentence_length > 0) {
        data_block->AddSentence(sentence, sentence_length,
          word_count, (uint64)rand() * 10000 + (uint64)rand());
      }
      else {
        //Reader read eof or has read data_block->size bytes before,
        //reader_->GetSentence will return 0
        delete[] sentence;
        break;
      }
    }

    multiverso::Log::Info("Rank %d LoadOneDataBlockTime:%lfs\n", process_id_,
      (clock() - start) / static_cast<double>(CLOCKS_PER_SEC));
  }

  //start the load data thread
  void DistributedWordembedding::StartLoadDataThread(Reader *reader, 
    int64 file_size) {
    for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch) {
      reader->ResetStart();
      for (int64 cur = 0; cur < file_size; cur += option_->data_block_size) {
        DataBlock *data_block = new (std::nothrow)DataBlock();
        assert(data_block != nullptr);
        LoadOneBlock(data_block, reader, option_->data_block_size);
        block_queue_->Push(data_block);

        //control the proload data
        while (static_cast<int64>(block_queue_->GetQueueSize()) *
          option_->data_block_size > option_->max_preload_data_size) {
          std::chrono::milliseconds dura(200);
          std::this_thread::sleep_for(dura);
        }
      }
    }

    DataBlock *data_block = new (std::nothrow)DataBlock();
    assert(data_block != nullptr);
    data_block->SetLastFlag();
    block_queue_->Push(data_block);
  }

  DataBlock* DistributedWordembedding::GetDataFromQueue() {
    DataBlock *temp = block_queue_->Pop();
    return temp;
  }

  DataBlock* DistributedWordembedding::GetBlockAndPrepareParameter() {
    DataBlock* data_block = GetDataFromQueue();
    if (data_block->Size() == 0) {
      return data_block;
    }
    data_block->MallocMemory(dictionary_->Size(), option_->use_adagrad);
    PrepareData(data_block);
    communicator_->RequestParameter(data_block);
    GetAllWordCount();
    return data_block;
  }

  void DistributedWordembedding::GetAllWordCount() {
    WordEmbedding_->word_count_actual = communicator_->GetWordCount();
    WordEmbedding_->UpdateLearningRate();
    multiverso::Log::Info("Get all word count done,word count actual is %lld\n",
      WordEmbedding_->word_count_actual);
  }

  void DistributedWordembedding::AddDeltaWordCount() {
    int64 temp_word_count = communicator_->GetWordCount();
    temp_word_count = WordEmbedding_->word_count_actual - temp_word_count;
    if (temp_word_count > 0) {
      communicator_->AddWordCount(temp_word_count);
      multiverso::Log::Info("Add word count done.word count delta is %lld\n",
        temp_word_count);
    }
  }

  void DistributedWordembedding::StartWordCount() {
    multiverso::Log::Info("Rank %d Start word count thread.\n", process_id_);
    int64 total_word_count = 0, sum = 0;
    while (is_running_) {
      sum = 0;
      for (int i = 0; i < trainers_.size(); ++i)
        sum += trainers_[i]->word_count;

      if (sum < 1000000 + total_word_count) {
        std::chrono::milliseconds dura(20);
        std::this_thread::sleep_for(dura);
      }
      else {
        WordEmbedding_->word_count_actual += sum - total_word_count;
        WordEmbedding_->UpdateLearningRate();
        total_word_count = sum;
        if (!option_->use_adagrad) {
          multiverso::Log::Info("Rank %d Alpha: %lf Progress: %.2lf%% \
                                WordCountActual:%lld Words/thread/second %lfk\n",
                                multiverso::MV_Rank(), WordEmbedding_->learning_rate,
                                WordEmbedding_->word_count_actual / (static_cast<double>\
                                (option_->total_words) * option_->epoch + 1) * 100,
                                WordEmbedding_->word_count_actual,
                                total_word_count / (static_cast<double>(option_->thread_cnt)\
                                * (clock() - start_) / CLOCKS_PER_SEC * 1000.0));
        }
        else {
          multiverso::Log::Info("Rank %d Progress: %.2lf%% WordCountActual: \
                                %lld Words/thread/second %lfk\n",
                                multiverso::MV_Rank(),
                                WordEmbedding_->word_count_actual / (static_cast<double>\
                                (option_->total_words) * option_->epoch + 1) * 100,
                                WordEmbedding_->word_count_actual,
                                total_word_count / (static_cast<double>(option_->thread_cnt)\
                                * (clock() - start_) / CLOCKS_PER_SEC * 1000.0));
        }
      }
    }

    //Add the left word_count to the WordEmbedding
    WordEmbedding_->word_count_actual += sum - total_word_count;
    WordEmbedding_->UpdateLearningRate();
  }

  void DistributedWordembedding::StartCollectWordcountThread() {
    is_running_ = true;
    collect_wordcount_thread_ = std::thread(
      &DistributedWordembedding::StartWordCount, this);
  }

  void DistributedWordembedding::StopCollectWordcountThread() {
    is_running_ = false;
    collect_wordcount_thread_.join();
  }

  void DistributedWordembedding::TrainNeuralNetwork() {
    int64 file_size = GetFileSize(option_->train_file);
    multiverso::Log::Info("train-file-size:%lld, data_block_size:%lld\n",
      file_size, option_->data_block_size);

    block_queue_ = new BlockQueue();
    load_data_thread_ = std::thread(&DistributedWordembedding::StartLoadDataThread,
      this, reader_, file_size);

    WordEmbedding_ = new WordEmbedding(option_, huffman_encoder_, sampler_,
      dictionary_->Size());
    assert(WordEmbedding_ != nullptr);

    for (int i = 0; i < option_->thread_cnt; ++i) {
      trainers_.push_back(new (std::nothrow) Trainer(i, option_, dictionary_,
        WordEmbedding_));
      assert(trainers_[i] != nullptr);
    }

    start_ = clock();
    int data_block_count = 0;
    DataBlock *next_block = nullptr;

    DataBlock *data_block = GetBlockAndPrepareParameter();
    if (data_block == nullptr) {
      multiverso::Log::Info("Please Change the Bigger Block Size.\n");
      return;
    }
    data_block_count++;

    StartCollectWordcountThread();
    for (int cur_epoch = 0; cur_epoch < option_->epoch; ++cur_epoch) {
      clock_t start_epoch = clock();
      for (int64 cur = 0; cur < file_size; cur += option_->data_block_size) {
        clock_t start_block = clock();

        //if don't use pipeline, training after getting parameters. 
        if (option_->is_pipeline == false) {
          #pragma omp parallel for num_threads(option_->thread_cnt)
          for (int i = 0; i < option_->thread_cnt; ++i) {
            trainers_[i]->TrainIteration(data_block);
          }

          multiverso::Log::Info("Rank %d Trainer training time:%lfs\n", process_id_,
            (clock() - start_block) / static_cast<double>(CLOCKS_PER_SEC));

          communicator_->AddDeltaParameter(data_block);
          AddDeltaWordCount();
          delete data_block;

          data_block = GetBlockAndPrepareParameter();
          data_block_count++;
        }
        //if use pipeline, training datablock and getting parameters of next
        //datablock in parallel. 
        else {
           #pragma omp parallel num_threads(option_->thread_cnt+1)
          {
            if (omp_get_thread_num() == option_->thread_cnt) {
              next_block = GetBlockAndPrepareParameter();
              data_block_count++;
            }
            else {
              trainers_[omp_get_thread_num()]->TrainIteration(data_block);
            }
          }

          multiverso::Log::Info("Rank %d Trainer training time:%lfs\n", process_id_,
            (clock() - start_block) / static_cast<double>(CLOCKS_PER_SEC));

          communicator_->AddDeltaParameter(data_block);
          AddDeltaWordCount();
          delete data_block;

          data_block = next_block;
          next_block = nullptr;
        }

        multiverso::Log::Info("Rank %d Dealing one block time:%lfs.\n", process_id_,
          (clock() - start_block) / static_cast<double>(CLOCKS_PER_SEC));
      }

      multiverso::Log::Info("Rank %d Dealing %d epoch time:%lfs\n", process_id_,
        cur_epoch, (clock() - start_epoch) / static_cast<double>(CLOCKS_PER_SEC));

      if (cur_epoch == option_->epoch - 1) {
        multiverso::MV_Barrier();
        if (process_id_ == 0) {
          SaveEmbedding(option_->output_file, option_->output_binary);
        }
      }
    }

    multiverso::Log::Info("Rank %d Finish Training %d Block.\n", process_id_,
      data_block_count);

    StopCollectWordcountThread();
    load_data_thread_.join();
    assert(data_block->isLast() == true);
    delete data_block;
    delete WordEmbedding_;
    delete block_queue_;
    for (auto trainer : trainers_) {
      delete trainer;
    }
  }
  //change the file name according to different epoch
  const char* DistributedWordembedding::ChangeFileName(const char *file_path,
    int iteration) {
    std::string temp(file_path);
    std::string c_iteration = temp + "_" + std::to_string(iteration);
    char * cstr = new char[c_iteration.length() + 1];
    std::strcpy(cstr, c_iteration.c_str());
    return cstr;
  }

  void DistributedWordembedding::SaveEmbedding(const char *file_path,
    bool is_binary) {
    multiverso::Log::Info("Rank %d Begin to Save Embeddings.\n", process_id_);

    clock_t start = clock();
    int epoch = dictionary_->Size() / kSaveBatch;
    int left = dictionary_->Size() % kSaveBatch;
    int base = 0;
    std::vector<real*> blocks;
    std::vector<int> nodes;

    FILE* fid = (is_binary == true) ? fid = fopen(file_path, "wb") :
      fid = fopen(file_path, "wt");
    fprintf(fid, "%d %d\n", dictionary_->Size(), option_->embeding_size);

    for (int i = 0; i < epoch; ++i) {
      for (int j = 0; j < kSaveBatch; ++j) {
        nodes.push_back(base + j);
      }

      memory_mamanger_->RequestBlocks(kSaveBatch, blocks);
      communicator_->GetWorkerTableRows(nodes, blocks, option_->embeding_size);
      WriteToFile(is_binary, blocks, fid, nodes);
      memory_mamanger_->ReturnBlocks(blocks);

      blocks.clear();
      nodes.clear();
      base = (i + 1)*kSaveBatch;
    }

    if (left > 0) {
      for (int j = 0; j < left; ++j) {
        nodes.push_back(base + j);
      }
      memory_mamanger_->RequestBlocks(left, blocks);
      communicator_->GetWorkerTableRows(nodes, blocks, option_->embeding_size);
      WriteToFile(is_binary, blocks, fid, nodes);
      memory_mamanger_->ReturnBlocks(blocks);
    }

    fclose(fid);
    multiverso::Log::Info("Rank %d Saving Embedding time:%lfs\n", process_id_,
      (clock() - start) / static_cast<double>(CLOCKS_PER_SEC));
  }

  void DistributedWordembedding::WriteToFile(bool is_binary,
    std::vector<real*> &blocks, FILE* fid, std::vector<int> &nodes){
    for (int i = 0; i < blocks.size(); ++i) {
      //get word id
      int id = nodes[i];
      fprintf(fid, "%s ", dictionary_->GetWordInfo(id)->word.c_str());
      for (int j = 0; j < option_->embeding_size; ++j) {
        if (is_binary) {
          real tmp = blocks[i][j];
          fwrite(&tmp, sizeof(real), 1, fid);
        }
        else {
          fprintf(fid, "%lf ", blocks[i][j]);
        }
      }
      fprintf(fid, "\n");
    }
  }

  void DistributedWordembedding::PrepareData(DataBlock *data_block) {
    clock_t start = clock();
    WordEmbedding_->PrepareData(data_block);
    multiverso::Log::Info("Rank %d Prepare data time:%lfs\n", process_id_,
      (clock() - start) / static_cast<double>(CLOCKS_PER_SEC));
  }

  void DistributedWordembedding::Train(int argc, char *argv[]) {
    argc = 1;
    argv = nullptr;
    multiverso::MV_Init(&argc, argv);
    multiverso::Log::Info("MV Rank %d Init done.\n", multiverso::MV_Rank());

    multiverso::MV_Barrier();
    multiverso::Log::Info("MV Barrier done.\n");
    //Mark the node machine number
    process_id_ = multiverso::MV_Rank();

    memory_mamanger_ = new MemoryManager(option_->embeding_size);
    assert(memory_mamanger_ != nullptr);
    communicator_ = new (std::nothrow)Communicator(option_, memory_mamanger_);
    assert(communicator_ != nullptr);
    //create worker table and server table
    communicator_->PrepareParameterTables(dictionary_->Size(),
      option_->embeding_size);

#ifdef _DEBUG
    multiverso::Log::ResetLogLevel(multiverso::LogLevel::Debug);
#endif
    //start to train
    TrainNeuralNetwork();

    multiverso::MV_ShutDown();
    multiverso::Log::Info("MV ShutDone done.\n");
    delete communicator_;
    delete memory_mamanger_;
  }

  void DistributedWordembedding::Run(int argc, char *argv[]) {
    g_log_suffix = GetSystemTime();
    srand(static_cast<unsigned int>(time(NULL)));

    option_ = new (std::nothrow)Option();
    assert(option_ != nullptr);

    dictionary_ = new (std::nothrow)Dictionary();
    assert(dictionary_ != nullptr);

    huffman_encoder_ = new (std::nothrow)HuffmanEncoder();
    assert(huffman_encoder_ != nullptr);

    //Parse argument and store them in option
    if (argc <= 1) {
      option_->PrintUsage();
      return;
    }

    option_->ParseArgs(argc, argv);

    //Read the vocabulary file; create the dictionary
    //and huffman_encoder according opt
    if ((option_->hs == 1) && (option_->negative_num != 0)) {
      multiverso::Log::Fatal
        ("The Hierarchical Softmax and Negative Sampling is indefinite!\n");
      exit(0);
    }

    option_->total_words = LoadVocab(option_, dictionary_, huffman_encoder_);
    option_->PrintArgs();

    sampler_ = new (std::nothrow)Sampler();
    assert(sampler_ != nullptr);
    if (option_->negative_num)
      sampler_->SetNegativeSamplingDistribution(dictionary_);

    char *filename = new (std::nothrow)char[strlen(option_->train_file) + 1];
    assert(filename != nullptr);
    strcpy(filename, option_->train_file);
    reader_ = new (std::nothrow)Reader(dictionary_, option_, sampler_, filename);
    assert(reader_ != nullptr);
    //Train with multiverso
    this->Train(argc, argv);

    delete option_;
    delete dictionary_;
    delete huffman_encoder_;
    delete sampler_;
    delete reader_;
    delete[]filename;
  }

  //Read the vocabulary file; create the dictionary
  //and huffman_encoder according opt
  int64 DistributedWordembedding::LoadVocab(Option *opt,
    Dictionary *dictionary, HuffmanEncoder *huffman_encoder) {
    int64 total_words = 0;
    char word[kMaxString];
    FILE* fid = nullptr;
    clock_t start = clock();
    multiverso::Log::Info("vocab_file %s\n", opt->read_vocab_file);

    if (opt->read_vocab_file != nullptr && strlen(opt->read_vocab_file) > 0) {
      multiverso::Log::Info("Begin to load vocabulary file [%s] ...\n",
        opt->read_vocab_file);
      fid = fopen(opt->read_vocab_file, "r");
      if (fid == nullptr) {
        multiverso::Log::Fatal("Open vocab_file failed!\n");
        exit(1);
      }
      int word_freq;
      while (fscanf(fid, "%s %d", word, &word_freq) != EOF) {
        dictionary->Insert(word, word_freq);
      }
    }

    dictionary->RemoveWordsLessThan(opt->min_count);
    multiverso::Log::Info("Dictionary size: %d\n", dictionary->Size());

    for (int i = 0; i < dictionary->Size(); ++i)
      total_words += dictionary->GetWordInfo(i)->freq;
    multiverso::Log::Info("Words in Dictionary %I64d\n", total_words);

    multiverso::Log::Info("Loading vocab time:%lfs\n",
      (clock() - start) / static_cast<double>(CLOCKS_PER_SEC));

    if (opt->hs)
      huffman_encoder->BuildFromTermFrequency(dictionary);
    if (fid != nullptr)
      fclose(fid);

    return total_words;
  }
}
