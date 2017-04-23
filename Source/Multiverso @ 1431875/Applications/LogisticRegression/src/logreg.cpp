#include "logreg.h"

#include <vector>
#include <sstream>

#include "util/log.h"
#include "util/common.h"
#include "reader.h"
#include "util/timer.h"
#include "multiverso/io/io.h"

namespace logreg {

template<typename EleType>
LogReg<EleType>::LogReg(const std::string &config_file) {
  config_ = new Configure(config_file);

  LR_CHECK((config_->input_size > 0) 
    && config_->read_buffer_size > 0
    && (!config_->sparse || config_->output_size > 0));
  // for delat
  config_->input_size += 1;

  if (config_->read_buffer_size % config_->minibatch_size != 0) {
    config_->read_buffer_size += config_->minibatch_size
      - (config_->read_buffer_size % config_->minibatch_size);
  }
  LR_CHECK(!config_->use_ps || !config_->pipeline || !config_->sparse || 
    (config_->read_buffer_size >= config_->minibatch_size * config_->sync_frequency));

  model_ = Model<EleType>::Get(*config_);
}

template<typename EleType>
LogReg<EleType>::~LogReg() {
  delete model_;
  delete config_;
}

template<typename EleType>
void LogReg<EleType>::Train(const std::string& train_file) {
  Log::Write(Info, "Train with file %s\n", train_file.c_str());

  int buffer_size = config_->read_buffer_size;

  auto reader = SampleReader<EleType>::Get(
    config_->reader_type,
    train_file, config_->input_size, 
    config_->output_size,
    config_->minibatch_size * config_->sync_frequency,
    buffer_size, config_->sparse);

  Sample<EleType>** samples = new Sample<EleType>*[buffer_size];

  int count = 0;
  int train_epoch = config_->train_epoch;
  size_t sample_seen = 0;
  float train_loss = 0.0f;
  size_t last = 0;
  for (int ep = 0; ep < train_epoch; ++ep) {
    reader->Reset();
    // wait for reading
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Log::Write(Info, "Start train epoch %d\n", ep);
    model_->SetKeys(reader->keys());
    do {
      while ((count = reader->Read(buffer_size, samples))) {
        Log::Write(Debug, "model training %d samples, sample seen %d\n", 
          count, sample_seen);
        train_loss += model_->Update(count, samples);
        sample_seen += count;
        if (sample_seen - last >= config_->show_time_per_sample) {
          Log::Write(Info, "Sample seen %lld, train loss %f\n", sample_seen, train_loss / (sample_seen - last));
          train_loss = 0.0f;
          last = sample_seen;
          model_->DisplayTime();
        }
        reader->Free(count);
      }
    } while (!reader->EndOfFile());
    Test();
  }
  delete reader;

  delete[]samples;
  Log::Write(Info, "Finish train, total sample %lld\n", sample_seen);
}

template<typename EleType>
void LogReg<EleType>::SaveModel() {
  SaveModel(config_->output_model_file);
}

template<typename EleType>
void LogReg<EleType>::SaveModel(const std::string& model_file) {
  Log::Write(Info, "Save model in file %s\n", model_file.c_str());
  model_->Store(model_file);
}

template<typename EleType>
void LogReg<EleType>::Train() {
  Train(config_->train_file);
}

template<typename EleType>
void SaveOutput(multiverso::Stream* stream, int num_output, 
  int output_size, EleType** output) {
  for (int i = 0; i < num_output; ++i) {
    std::stringstream ss;
    EleType* line = output[i];
    ss << line[0];
    for (int j = 1; j < output_size; ++j) {
      ss << ' ' << line[j];
    }
    ss << '\n';
    stream->Write(ss.str().c_str(), ss.str().length());
  }
}

template<typename EleType>
double LogReg<EleType>::Test(const std::string& test_file, EleType**result) {
  if (test_file == "") {
    return 0.0;
  }
  Log::Write(Info, "Test with file %s\n", test_file.c_str());

  int buffer_size = config_->read_buffer_size;
  auto reader = SampleReader<EleType>::Get(
    config_->reader_type,
    test_file, config_->input_size,
    config_->output_size,
    config_->minibatch_size * config_->sync_frequency,
    buffer_size, config_->sparse);
  
  reader->Reset();

  bool own_result = false;
  if (result == nullptr) {
    result = CreateMatrix<EleType>(buffer_size,
      config_->output_size);
    own_result = true;
  }

  Sample<EleType>** samples = new Sample<EleType>*[buffer_size];

  auto stream = multiverso::StreamFactory::GetStream(
    multiverso::URI(config_->output_file), 
    multiverso::FileOpenMode::Write);

  size_t correct_count = 0;
  size_t total_sample = 0;
  model_->SetKeys(reader->keys());
  do {
    int count;
    while ((count = reader->Read(buffer_size, samples))) {
      total_sample += count;
      correct_count += model_->Predict(count, samples, result);
      reader->Free(count);
      SaveOutput(stream, count, config_->output_size, result);
    }
  } while (!reader->EndOfFile());
  
  double test_error = 1.0 - static_cast<double>(correct_count) / total_sample;
  Log::Write(Info, "test error: %f\n", test_error);

  delete reader;
  delete stream;
  delete[]samples;
  if (own_result) {
    FreeMatrix(buffer_size, result);
  }
  return test_error;
}

template<typename EleType>
double LogReg<EleType>::Test(EleType**result) {
  return Test(config_->test_file, result);
}

template<typename EleType>
DataBlock<EleType>* LogReg<EleType>::model() const {
  return model_->table();
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(LogReg);

}  // namespace logreg
