// defines SwigBoost::shared_ptr, a wrapper around boost::shared_ptr
// Use this shared_ptr wrapper for testing memory leaks of shared_ptr.
// getTotalCount() should return zero at end of test

#include <iostream>

struct SWIG_null_deleter; // forward reference, definition is in shared_ptr.i
namespace SwigBoost {
// This template can be specialized for better debugging information
template <typename T> std::string show_message(boost::shared_ptr<T>*t) {
  if (!t)
    return "null shared_ptr!!!";
  if (boost::get_deleter<SWIG_null_deleter>(*t))
    return std::string(typeid(t).name()) + " NULL DELETER";
  if (*t)
    return std::string(typeid(t).name()) + " object";
  else
    return std::string(typeid(t).name()) + " NULL";
}

namespace SharedPtrWrapper {
  static SwigExamples::CriticalSection critical_section;
  static int total_count = 0;

  template<typename T> void increment(boost::shared_ptr<T>* ptr) { 
    SwigExamples::Lock lock(critical_section); 
    std::cout << "====SharedPtrWrapper==== + " << ptr << " " << show_message(ptr) << " " <<  std::endl << std::flush;
    total_count++;
  }
  template<typename T> void decrement(boost::shared_ptr<T>* ptr) {
    SwigExamples::Lock lock(critical_section); 
    std::cout << "====SharedPtrWrapper==== - " << ptr << " " << show_message(ptr) << " " <<  std::endl << std::flush;
    total_count--;
  }
  static int getTotalCount() { return total_count; }
}

template<typename T> class shared_ptr {
private:
    typedef shared_ptr<T> this_type;
public:
    typedef typename boost::detail::shared_ptr_traits<T>::reference reference;

  shared_ptr() : m_shared_ptr() {
    SharedPtrWrapper::increment(&m_shared_ptr);
  }
  template<typename Y> explicit shared_ptr(Y* p) : m_shared_ptr(p) {
    SharedPtrWrapper::increment(&m_shared_ptr);
  }
  template<typename Y, typename D> explicit shared_ptr(Y* p, D d) : m_shared_ptr(p, d) {
    SharedPtrWrapper::increment(&m_shared_ptr);
  }

  shared_ptr(shared_ptr const & other)
    : m_shared_ptr(other.m_shared_ptr)
  {
    SharedPtrWrapper::increment(&m_shared_ptr);
  }

  template<typename Y> shared_ptr(shared_ptr<Y> const & other)
    : m_shared_ptr(other.m_shared_ptr)
  {
    SharedPtrWrapper::increment(&m_shared_ptr);
  }

  reference operator*() const {
    return m_shared_ptr.operator*();
  }
  T* operator->() const {
    return m_shared_ptr.operator->();
  }
  T* get() const { 
    return m_shared_ptr.get();
  }
  operator bool() const {
    return m_shared_ptr.get() == 0 ? false : true;
  }
  bool unique() const {
    return m_shared_ptr.unique();
  }
  long use_count() const {
    return m_shared_ptr.use_count();
  }
  void swap(shared_ptr<T>& other) {
    std::swap(m_shared_ptr, other.m_shared_ptr);
  }
  template<class Y> bool _internal_less(shared_ptr<Y> const & rhs) const {
    return m_shared_ptr < rhs.m_shared_ptr;
  }
  ~shared_ptr() {
    SharedPtrWrapper::decrement(&m_shared_ptr);
  }

private:
  template<class Y> friend class shared_ptr;

  boost::shared_ptr<T> m_shared_ptr;
};
}

