
# if defined(_WIN32) || defined(__WIN32__)

#include <windows.h>

namespace SwigExamples {

class CriticalSection {
public:
  CriticalSection() {
    InitializeCriticalSection(&mutex_);
  }
  ~CriticalSection() {
    DeleteCriticalSection(&mutex_);
  }
  CRITICAL_SECTION mutex_;
};

struct Lock {
  Lock(CriticalSection &cs) : critical_section(cs) {
    EnterCriticalSection(&critical_section.mutex_);
  }
  ~Lock() {
    LeaveCriticalSection(&critical_section.mutex_);
  }
private:
  CriticalSection &critical_section;
};

}

#else

#include <pthread.h>
#ifndef PTHREAD_MUTEX_RECURSIVE_NP
  // For Cygwin and possibly other OSs: _NP is "non-portable"
  #define PTHREAD_MUTEX_RECURSIVE_NP PTHREAD_MUTEX_RECURSIVE
#endif

namespace SwigExamples {

class CriticalSection {
public:
  CriticalSection() {
    pthread_mutexattr_t mutexattr;
    pthread_mutexattr_init(&mutexattr);
    pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_RECURSIVE_NP);
    pthread_mutex_init(&mutex_, &mutexattr);
    pthread_mutexattr_destroy(&mutexattr);
  }
  ~CriticalSection() {
    pthread_mutex_destroy (&mutex_);
  }
  pthread_mutex_t mutex_;
};

struct Lock {
  Lock(CriticalSection &cs) : critical_section(cs) {
    pthread_mutex_lock (&critical_section.mutex_);
  }
  ~Lock() {
    pthread_mutex_unlock (&critical_section.mutex_);
  }
private:
  CriticalSection &critical_section;
};

}

#endif

