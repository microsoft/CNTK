/* File : example.h */

// A simple exception
class EmptyError { };
class FullError { 
 public:
  int maxsize;
  FullError(int m) : maxsize(m) { }
};

template<typename T> class Queue {
  int maxsize;
  T   *items;
  int nitems;
  int last;
 public: 
  Queue(int size) {
    maxsize = size;
    items = new T[size];
    nitems = 0;
    last = 0;
  }
  ~Queue() {
    delete [] items;
  }
  void enqueue(T x) throw(FullError) {
    if (nitems == maxsize) {
      throw FullError(maxsize);
    }
    items[last] = x;
    last = (last + 1) % maxsize;
    nitems++;
  }
  T dequeue()  {
    T x;
    if (nitems == 0) throw EmptyError();
    x = items[(last + maxsize - nitems) % maxsize];
    nitems--;
    return x;
  }
  int length() {
    return nitems;
  }
};






  
