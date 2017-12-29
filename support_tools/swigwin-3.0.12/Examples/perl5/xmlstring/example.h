#include <xercesc/util/XMLString.hpp>


class XMLChTest 
{
  XMLCh *_val;
  
public:

  XMLChTest() : _val(0)
  {
  }

  void set(const XMLCh *v) 
  {
    size_t len = XERCES_CPP_NAMESPACE::XMLString::stringLen(v);
    delete[] _val;
    _val = new XMLCh[len + 1];
    for (int i = 0; i < len; ++i) {
      _val[i] = v[i];
    }
    _val[len] = 0;
  }

  const XMLCh *get() 
  {
    return _val;
  }

  XMLCh get_first() 
  {
    return _val[0];
  }
  
};

