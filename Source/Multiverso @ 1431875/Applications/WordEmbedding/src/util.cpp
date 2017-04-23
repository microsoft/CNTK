#include <time.h>
#include "util.h"

namespace wordembedding {

  Option::Option() {
    train_file = nullptr;
    read_vocab_file = nullptr;
    output_file = nullptr;
    sw_file = nullptr;
    endpoints_file = "";
    hs = false;
    negative_num = 5;
    output_binary = false;
    sample = 0;
    cbow = true;
    embeding_size = 100;
    thread_cnt = 1;
    window_size = 5;
    min_count = 5;
    data_block_size = 1000000;
    init_learning_rate = static_cast<real>(0.025);
    epoch = 1;
    stopwords = false;
    is_pipeline = true;
    total_words = 0;
    max_preload_data_size = 8000000000LL;
    use_adagrad = false;
  }
  //Input all the local model-arguments 
  void Option::ParseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i += 2) {
      if (strcmp(argv[i], "-size") == 0) embeding_size = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-train_file") == 0) train_file = argv[i + 1];
      if (strcmp(argv[i], "-endpoints_file") == 0) endpoints_file = argv[i + 1];
      if (strcmp(argv[i], "-read_vocab") == 0) read_vocab_file = argv[i + 1];
      if (strcmp(argv[i], "-binary") == 0) output_binary = (atoi(argv[i + 1]) != 0);
      if (strcmp(argv[i], "-cbow") == 0) cbow = (atoi(argv[i + 1]) != 0);
      if (strcmp(argv[i], "-alpha") == 0) init_learning_rate = static_cast<real>(atof(argv[i + 1]));
      if (strcmp(argv[i], "-output") == 0) output_file = argv[i + 1];
      if (strcmp(argv[i], "-window") == 0) window_size = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-sample") == 0) sample = static_cast<real>(atof(argv[i + 1]));
      if (strcmp(argv[i], "-hs") == 0) hs = (atoi(argv[i + 1]) != 0);
      if (strcmp(argv[i], "-data_block_size") == 0) data_block_size = atoll(argv[i + 1]);
      if (strcmp(argv[i], "-max_preload_data_size") == 0) max_preload_data_size = atoll(argv[i + 1]);
      if (strcmp(argv[i], "-negative") == 0) negative_num = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-threads") == 0) thread_cnt = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-min_count") == 0) min_count = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-epoch") == 0) epoch = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-stopwords") == 0) stopwords = (atoi(argv[i + 1]) != 0);
      if (strcmp(argv[i], "-sw_file") == 0)  sw_file = argv[i + 1];
      if (strcmp(argv[i], "-use_adagrad") == 0) use_adagrad = (atoi(argv[i + 1]) != 0);
      if (strcmp(argv[i], "-is_pipeline") == 0) is_pipeline = (atoi(argv[i + 1]) != 0);

    }
  }

  void Option::PrintUsage() {
    puts("Usage:");
    puts("-size: word embedding size, e.g. 300");
    puts("-train_file: the training corpus file, e.g.enwik2014");
    puts("-read_vocab : the file to read all the vocab counts info");
    puts("-binary : 0 or 1, indicates whether to write all the embeddings vectors into binary format");
    puts("-cbow : 0 or 1, default 1, whether to use cbow or not");
    puts("-alpha : initial learning rate, usually set to 0.025");
    puts("-output : the output file to store all the embedding vectors");
    puts("-window : the window size");
    puts("-sample : the sub - sample size, usually set to 0");
    puts("-hs : 0 or 1, default 1, whether to use hierarchical softmax");
    puts("-negative : the negative word count in negative sampling, please set it to 0 when - hs = 1");
    puts("-threads : the thread number to run in one machine");
    puts("-min_count : words with lower frequency than min_count is removed from dictionary");
    puts("-epoch : the epoch number");
    puts("-stopwords : 0 or 1, whether to avoid training stop words");
    puts("-sw_file : the stop words file storing all the stop words, valid when -stopwords = 1");
    puts("-use_adagrad : 0 or 1, whether to use adagrad to adjust learnin rate");
    puts("-data_block_size : default 1MB, the maximum bytes which a data block will store");
    puts("-max_preload_data_size : default 8GB, the maximum data size(bytes) which multiverse_WordEmbedding will preload");
    puts("-is_pipeline : 0 or 1, whether to use pipeline");
    puts("-server_endpoint_file : default "", server ZMQ socket endpoint file in MPI - free version");
  }

  void Option::PrintArgs() {
    multiverso::Log::Info("train_file: %s\n", train_file);
    multiverso::Log::Info("read_vocab_file: %s\n", read_vocab_file);
    multiverso::Log::Info("output_file: %s\n", output_file);
    multiverso::Log::Info("sw_file: %s\n", sw_file);
    multiverso::Log::Info("hs: %d\n", hs);
    multiverso::Log::Info("output_binary: %d\n", output_binary);
    multiverso::Log::Info("cbow: %d\n", cbow);
    multiverso::Log::Info("stopwords: %d\n", stopwords);
    multiverso::Log::Info("use_adagrad: %d\n", use_adagrad);
    multiverso::Log::Info("sample: %lf\n", sample);
    multiverso::Log::Info("embeding_size: %d\n", embeding_size);
    multiverso::Log::Info("thread_cnt: %d\n", thread_cnt);
    multiverso::Log::Info("window_size: %d\n", window_size);
    multiverso::Log::Info("negative_num: %d\n", negative_num);
    multiverso::Log::Info("min_count: %d\n", min_count);
    multiverso::Log::Info("epoch: %d\n", epoch);
    multiverso::Log::Info("total_words: %lld\n", total_words);
    multiverso::Log::Info("max_preload_data_size: %lld\n", max_preload_data_size);
    multiverso::Log::Info("init_learning_rate: %lf\n", init_learning_rate);
    multiverso::Log::Info("data_block_size: %lld\n", data_block_size);
    multiverso::Log::Info("is_pipeline: %d\n", is_pipeline);
    multiverso::Log::Info("endpoints_file: %s\n", endpoints_file);
  }

  Sampler::Sampler() {
    table_ = nullptr;
  }

  Sampler::~Sampler() {
    free(table_);
  }
  //Set the negative-sampling distribution
  void Sampler::SetNegativeSamplingDistribution(Dictionary *dictionary) {
    real train_words_pow = 0;
    real power = 0.75;
    table_ = (int *)malloc(kTableSize * sizeof(int));
    for (int i = 0; i < dictionary->Size(); ++i)
      train_words_pow += static_cast<real>(pow(dictionary->GetWordInfo(i)->freq, power));
    int cur_pos = 0;
    real d1 = (real)pow(dictionary->GetWordInfo(cur_pos)->freq, power)
      / (real)train_words_pow;

    assert(table_ != nullptr);
    for (int i = 0; i < kTableSize; ++i) {
      table_[i] = cur_pos;
      if (i > d1 * kTableSize && cur_pos + 1 < dictionary->Size()) {
        cur_pos++;
        d1 += (real)pow(dictionary->GetWordInfo(cur_pos)->freq, power)
          / (real)train_words_pow;
      }
    }
  }

  bool Sampler::WordSampling(int64 word_cnt,
    int64 train_words, real sample) {
    real ran = (sqrt(word_cnt / (sample * train_words)) + 1) *
      (sample * train_words) / word_cnt;
    return (ran > ((real)rand() / (RAND_MAX)));
  }
  //Get the next random 
  uint64 Sampler::GetNextRandom(uint64 next_random) {
    return next_random * (uint64)25214903917 + 11;
  }

  int Sampler::NegativeSampling(uint64 next_random) {
    return table_[(next_random >> 16) % kTableSize];
  }

  std::string GetSystemTime() {
    time_t t = time(0);
    char tmp[128];
    strftime(tmp, sizeof(tmp), "%Y%m%d%H%M%S", localtime(&t));
    return std::string(tmp);
  }

  //Get the size of filename, it should deal with large files
  int64 GetFileSize(const char *filename) {
#ifdef _MSC_VER
    struct _stat64 info;
    _stat64(filename, &info);
    return (int64)info.st_size;
#else
    struct  stat info;
    stat(filename, &info);
    return(int64)info.st_size;
#endif  
  }

  //Readword from train_file to word array by the word index
  bool ReadWord(char *word, FILE *fin) {
    int idx = 0;
    char ch;
    while (!feof(fin)) {
      ch = fgetc(fin);
      if (ch == 13) continue;
      if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
        if (idx > 0) {
          if (ch == '\n')
            ungetc(ch, fin);
          break;
        }

        if (ch == '\n') {
          strcpy(word, (char *)"</s>");
          return true;
        }
        else {
          continue;
        }
      }

      word[idx++] = ch;
      if (idx >= kMaxString - 1)
        idx--;
    }

    word[idx] = 0;
    return idx > 0;
  }

  std::string g_log_suffix;
  real* expTable;
  int embedding_size;

  void InitExpTable() {
    expTable = (real *)malloc((kExpTableSize + 1) * sizeof(real));
    for (int i = 0; i < kExpTableSize; i++) {
      // Precompute the exp() table
      expTable[i] = exp((i / (real)kExpTableSize * 2 - 1) * kMaxExp);
      // Precompute f(x) = x / (x + 1)
      expTable[i] = expTable[i] / (expTable[i] + 1);
    }
  }
}