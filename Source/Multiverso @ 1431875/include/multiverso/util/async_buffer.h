#ifndef  MULTIVERSO_UTIL_ASYNC_BUFFER_H_
#define  MULTIVERSO_UTIL_ASYNC_BUFFER_H_

#include <multiverso/table_interface.h>
#include <multiverso/util/waiter.h>
#include <thread>

namespace multiverso {

template<typename BufferType = void*>
class ASyncBuffer{
public:
  // Creates an async buffer
  // buffer1 and buffer2: buffers used to save data from server
  // fill_buffer_action: action to fill a given buffer.
  ASyncBuffer(BufferType* buffer0, BufferType* buffer1,
    std::function<void(BufferType*)> fill_buffer_action)
    : buffer_writer_{ fill_buffer_action } {
    CHECK_NOTNULL(buffer0);
    CHECK_NOTNULL(buffer1);
    //[TODO(qiwye)] to make buffer number configurable.
    buffers_.resize(2);
    buffers_[0] = buffer0;
    buffers_[1] = buffer1;
    Init();
  }

  // Returns the ready buffer.
  // This function also automatically starts to prefetch data
  //  for the other buffer.
  BufferType* Get() {
    if (thread_ == nullptr) {
      Init();
    }

    ready_waiter_.Wait();
    auto ready_buffer = WritableBuffer(current_task_);
    PrefetchNext();
    return ready_buffer;
  }

  ~ASyncBuffer() {
    if (thread_ != nullptr) {
      Join();
    }
  }


  // Stops prefetch and releases related resource
  void Join() {
    if (thread_ != nullptr) {
      ready_waiter_.Wait();
      current_task_ = STOP_THREAD;
      new_task_waiter_.Notify();
      if (thread_->joinable()) {
        thread_->join();
      }
      thread_ = nullptr;
    }
  }
private:
  ASyncBuffer(const ASyncBuffer<BufferType>&);
  ASyncBuffer& operator = (const ASyncBuffer<BufferType>&);

protected:
  enum TaskType {
    FILL_BUFFER0,
    FILL_BUFFER1,
    STOP_THREAD
  };

protected:
  void Init() {
    ready_waiter_.Reset(0);
    new_task_waiter_.Reset(1);
    current_task_ = FILL_BUFFER1;
    thread_ =
      new std::thread(&ASyncBuffer<BufferType>::fill_buffer_routine,
      this);
    PrefetchNext();
  }

  void PrefetchNext() {
    current_task_ = (current_task_ == FILL_BUFFER1) ?
    FILL_BUFFER0 : FILL_BUFFER1;
    ready_waiter_.Reset(1);
    new_task_waiter_.Notify();
  }

  BufferType* WritableBuffer(TaskType task) {
    CHECK(task != STOP_THREAD);
    return task == FILL_BUFFER0 ? buffers_[0] : buffers_[1];
  }

private:
  std::vector<BufferType*> buffers_;
  std::function<void(BufferType*)> buffer_writer_;
  Waiter ready_waiter_;
  Waiter new_task_waiter_;
  TaskType current_task_;
  std::thread * thread_;

private:
  void fill_buffer_routine() {
    while (true) {
      new_task_waiter_.Wait();
      if (current_task_ == STOP_THREAD) {
        break;
      }

      buffer_writer_(WritableBuffer(current_task_));
      ready_waiter_.Notify();
      new_task_waiter_.Reset(1);
    }
  }
};

}  // namespace multiverso


#endif  // MULTIVERSO_UTIL_ASYNC_BUFFER_H_
