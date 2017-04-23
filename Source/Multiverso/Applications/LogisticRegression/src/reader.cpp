#include "reader.h"

#include <algorithm>
#include <sstream>

#include "util/common.h"
#include "util/log.h"
#include "util/timer.h"

namespace logreg {

template <typename EleType>
inline EleType min(EleType a, EleType b) {
  return a < b ? a : b;
}

template <typename EleType>
SampleReader<EleType>::SampleReader(
  const std::string& files,
  size_t row_size,
  int output_size,
  size_t update_per_sample,
  int max_row_buffer_count,
  bool sparse,
  bool init) :
    sparse_(sparse),
    eof_(true),
    reader_(nullptr),
    reading_file_(0),
    thread_(nullptr),
    row_size_(row_size),
    output_size_(output_size),
    // use 2x size buffer
    buffer_size_(max_row_buffer_count * 3),
    sample_batch_size_(update_per_sample),
    sample_count_(0) {
  // parse files
  size_t p = files.find(';');
  size_t prev = 0;
  while (p != -1) {
    files_.push_back(files.substr(prev, p - prev));
    prev = p + 1;
    p = files.find(';', prev);
  }
  files_.push_back(files.substr(prev));

  buffer_ = CeateSamples<EleType>(buffer_size_, row_size, sparse);


  if (init) {
    thread_ = new std::thread(&SampleReader<EleType>::Main, this);
    Log::Write(Debug, "Init SampleReader, files %s, buffer size %d\n",
      files.c_str(), buffer_size_);
  }

  cur_keys_ = new SparseBlock<bool>();
}

template <typename EleType>
SampleReader<EleType>::~SampleReader() {
  FreeSamples(buffer_size_, buffer_);

  delete cur_keys_;
  DeleteKeys();

  buffer_size_ = 0;
  thread_->join();
  delete thread_;
  delete reader_;
}

template <typename EleType>
void SampleReader<EleType>::Reset() {
  DEBUG_CHECK(eof_);
  reading_file_ = 0;
  delete reader_;
  reader_ = new TextReader(URI(files_[0]), 1024);
  DeleteKeys();
  start_ = end_ = 0;
  length_ = read_length_ = 0;
  eof_ = false;
  Log::Write(Debug, "SampleReader reset\n");
}

inline int round(int cur, int size) {
  return (cur >= size) ? (cur - size) : cur;
}

template <typename EleType>
int SampleReader<EleType>::Read(int num_row, Sample<EleType>**buffer) {
  int size;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    int length = length_ - read_length_;
    size = min<int>(length, num_row);
    read_length_ += size;
    // Log::Write(Debug, "SampleReader read [%d / %d] rows. length: %d, \
      // read_length: %d\n", size, num_row, length_, read_length_);
  }
  for (int i = 0; i < size; ++i) {
    buffer[i] = buffer_[round(start_ + i, buffer_size_)];
  }
  start_ = round(start_ + size, buffer_size_);
  return size;
}

template <typename EleType>
int SampleReader<EleType>::Ask() {
  std::lock_guard<std::mutex> lock(mutex_);
  return length_ - read_length_;
}

template <typename EleType>
void SampleReader<EleType>::Free(int num_row) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    length_ -= num_row;
    read_length_ -= num_row;
  }
  // timer_.Start();
  cv_.notify_all();
  // Log::Write(Debug, "SampleReader free %d rows. length: %d, \
    // read_length: %d\n", num_row, length_, read_length_);
}

template <typename EleType>
void SampleReader<EleType>::Main() {
  std::string line;
  Log::Write(Debug, "Start reader thread\n");
  while (true) {
    while (eof_) {
      if (!buffer_size_)  return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Log::Write(Debug, "SampleReader begin read file %s\n", 
      files_[reading_file_].c_str());
    std::unique_lock<std::mutex> lock(mutex_);
    // timer_.Start();
    while (reader_->GetLine(line)) {
      /* if (length_ == buffer_size_) {
        Log::Write(Debug, "reader read sample average time %f\n", \
          timer_.ElapseMilliSeconds() / buffer_size_ * 2);
      } */
      while (length_ == buffer_size_) {
        cv_.wait(lock);
      }
      ParseLine(line, end_);
      ++length_;
      end_ = round(++end_, buffer_size_);
    }
    Log::Write(Debug, "SampleReader end of file %s\n", 
      files_[reading_file_++].c_str());
    if (reading_file_ < files_.size()) {
      delete reader_;
      reader_ = new TextReader(URI(files_[reading_file_]), 1024);
      continue;
    }
    if (cur_keys_->size() != 0) {
      keys_.Push(cur_keys_);
      cur_keys_ = new SparseBlock<bool>();
    }
    eof_ = true;
  }
  Log::Write(Debug, "End reader thread\n");
}

template<typename EleType>
void SampleReader<EleType>::DeleteKeys() {
  SparseBlock<bool> *tmp;
  while (!keys_.Empty()) {
    keys_.Pop(tmp);
    delete tmp;
  }
}

template<typename EleType>
void SampleReader<EleType>::ParseLine(
  const std::string&line, int idx) {
  // Log::Write(Debug, "SampleReader Parse Line\n");
  Sample<EleType>* data = this->buffer_[idx];
  data->values.clear();

  std::stringstream iss(line);
  iss >> data->label;
  
  size_t index;
  EleType value;
  if (sparse_) {
    data->keys.clear();
    char c;
    while (iss >> index >> c >> value) {
      data->keys.push_back(index); 
      data->values.push_back(value);
    }
    // for bias
    data->keys.push_back(row_size_ - 1);
    for (int i = 0; i < data->keys.size(); ++i) {
      index = data->keys[i];
      for (int j = 0; j < output_size_; ++j) {
        cur_keys_->Set(index, true);
        index += row_size_;
      }
    }
    if (++sample_count_ == sample_batch_size_) {
      sample_count_ = 0;
      keys_.Push(cur_keys_);
      cur_keys_ = new SparseBlock<bool>();
    }
  } else {
    index = 0;
    while (iss >> value) {
      data->values.push_back(value);
    }
  }
  // for bias
  data->values.push_back(1);
}

template<typename EleType>
SampleReader<EleType>* SampleReader<EleType>::Get(
  const std::string&type,
  const std::string&files,
  size_t row_size,
  int output_size,
  size_t update_per_sample,
  int max_row_buffer_count,
  bool sparse) {
  if (type == "weight") {
    return new WeightedSampleReader<EleType>(files, row_size, output_size,
      update_per_sample, max_row_buffer_count, sparse);
  } else if (type == "bsparse") {
    return new BSparseSampleReader<EleType>(files, row_size, output_size,
      update_per_sample, max_row_buffer_count, sparse);
  }
  // default
  return new SampleReader<EleType>(files, row_size, output_size,
    update_per_sample, max_row_buffer_count, sparse);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(SampleReader);

template<typename EleType>
void WeightedSampleReader<EleType>::ParseLine(
  const std::string&line, int idx) {
  // Log::Write(Debug, "SampleReader Parse Line\n");
  Sample<EleType>* data = this->buffer_[idx];
  data->values.clear();

  std::stringstream iss(line);
  
  char c;
  double weight;
  iss >> data->label >> c >> weight;

  size_t index;
  EleType value;
  if (this->sparse_) {
    data->keys.clear();
    while (iss >> index >> c >> value) {
      data->keys.push_back(index);
      data->values.push_back((EleType)(value * weight));
      for (int i = 0; i < this->output_size_; ++i) {
        this->cur_keys_->Set(index, true);
        index += this->row_size_;
      }
    }
    // for bias
    data->keys.push_back(this->row_size_ - 1);
    index = this->row_size_ - 1;
    for (int i = 0; i < this->output_size_; ++i) {
      this->cur_keys_->Set(index, true);
      index += this->row_size_;
    }
    if (++this->sample_count_ == this->sample_batch_size_) {
      this->sample_count_ = 0;
      this->keys_.Push(this->cur_keys_);
      this->cur_keys_ = new SparseBlock<bool>();
    }
  } else {
    while (iss >> value) {
      data->values.push_back((EleType)(value * weight));
    }
  }
  // for bias
  data->values.push_back(1);
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(WeightedSampleReader);

template<typename EleType>
BSparseSampleReader<EleType>::BSparseSampleReader(
  const std::string&files,
  size_t row_size,
  int output_size,
  size_t update_per_sample,
  int max_row_buffer_count,
  bool sparse) :
  SampleReader<EleType>(files, row_size, output_size, update_per_sample,
    max_row_buffer_count, sparse, false),
  stream_(nullptr),
  chunk_idx_(0),
  chunk_size_(0) {
    LR_CHECK(sparse);

    this->thread_ = new std::thread(&BSparseSampleReader<EleType>::Main, this);
    Log::Write(Debug, "Init BSparseSampleReader, files %s, buffer size %d\n",
      files.c_str(), this->buffer_size_);

    data_chunk_.resize(chunk_capacity_);
}

template<typename EleType>
BSparseSampleReader<EleType>::~BSparseSampleReader() {
  delete this->stream_;
}

template <typename EleType>
void BSparseSampleReader<EleType>::Reset() {
  DEBUG_CHECK(this->eof_);
  this->reading_file_ = 0;
  delete stream_;
  this->stream_ = multiverso::StreamFactory::GetStream(URI(this->files_[0]), FileOpenMode::BinaryRead);
  this->start_ = this->end_ = 0;
  this->length_ = this->read_length_ = 0;
  chunk_idx_ = 0;
  chunk_size_ = 0;
  this->DeleteKeys();
  this->eof_ = false;
  Log::Write(Debug, "BSparseSampleReader reset\n");
}

template <typename EleType>
void BSparseSampleReader<EleType>::Main() {
  Log::Write(Debug, "Start reader thread\n");
  while (true) {
    while (this->eof_) {
      if (!this->buffer_size_)  return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Log::Write(Debug, "BSparseSampleReader begin read file %s\n", 
      this->files_[this->reading_file_].c_str());
    std::unique_lock<std::mutex> lock(this->mutex_);
    // timer_.Start();
    while (true) {
      /*if (length_ == buffer_size_) {
        Log::Write(Debug, "reader read sample average time %f\n", timer_.ElapseMilliSeconds() / buffer_size_ * 2);
      }*/
      while (this->length_ == this->buffer_size_) {
        this->cv_.wait(lock);
      }
      if (!ParseSample(this->end_)) {
        break;
      }
      ++this->length_;
      this->end_ = round(++this->end_, this->buffer_size_);
    }
    Log::Write(Debug, "BSparseSampleReader end of file %s\n", 
      this->files_[this->reading_file_++].c_str());
    if (this->reading_file_ < this->files_.size()) {
      delete this->stream_;
      this->stream_ = StreamFactory::GetStream(
        URI(this->files_[this->reading_file_]), FileOpenMode::BinaryRead);
      continue;
    }
    if (this->cur_keys_->size() != 0) {
      this->keys_.Push(this->cur_keys_);
      this->cur_keys_ = new SparseBlock<bool>();
    }
    this->eof_ = true;
  }
  Log::Write(Debug, "End reader thread\n");
}


template <typename EleType>
int BSparseSampleReader<EleType>::LoadDataChunk() {
  char* buf = data_chunk_.data();
  chunk_size_ -= chunk_idx_;
  if (chunk_size_ != 0) {
    memcpy(buf, buf + chunk_idx_, chunk_size_);
    buf += chunk_size_;
  }
  
  chunk_size_ += static_cast<int>(stream_->Read(buf, 
    chunk_capacity_ - chunk_size_));
  chunk_idx_ = 0;
  return chunk_size_;
}

template <typename EleType>
bool BSparseSampleReader<EleType>::ParseSample(int idx) {
  static const int head = static_cast<int>(
    sizeof(size_t)+sizeof(int)+sizeof(double));

  if (chunk_size_ - chunk_idx_ < head && LoadDataChunk() == 0) {
    return false;
  }

  char* buf = data_chunk_.data();
  Sample<EleType>* data = this->buffer_[idx];

  size_t size = *reinterpret_cast<size_t*>(buf + chunk_idx_);
  chunk_idx_ += sizeof(size_t);
  data->keys.resize(size + 1);
  data->values.resize(size + 1);

  data->label = *reinterpret_cast<int*>(buf + chunk_idx_);
  chunk_idx_ += sizeof(int);

  double weight = *reinterpret_cast<double*>(buf + chunk_idx_);
  chunk_idx_ += sizeof(double);

  if (chunk_size_ - chunk_idx_ < static_cast<int>(sizeof(size_t)* size)) {
    LoadDataChunk();
    buf = data_chunk_.data();
  }
  buf += chunk_idx_;

  memcpy(data->keys.data(), buf, sizeof(size_t)* size);
  chunk_idx_ += static_cast<int>(sizeof(size_t)* size);
  data->keys[size] = this->row_size_ - 1;  // bias

  std::fill_n(data->values.data(), size + 1, (EleType)weight);

  for (int i = 0; i <= size; ++i) {
    size_t index = data->keys[i];
    for (int j = 0; j < this->output_size_; ++j) {
      this->cur_keys_->Set(index, true);
      index += this->row_size_;
    }
  }
  if (++this->sample_count_ == this->sample_batch_size_) {
    this->sample_count_ = 0;
    this->keys_.Push(this->cur_keys_);
    this->cur_keys_ = new SparseBlock<bool>();
  }
  return true;
}

DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(BSparseSampleReader);

}  // namespace logreg
