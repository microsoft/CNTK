// bool typemaps check
%module bools

#if defined(SWIGSCILAB)
%rename(BoolSt) BoolStructure;
#endif

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);                   /* memory leak when setting a ptr/ref variable */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) constbool;         /* Ruby, wrong class name */

// bool constant
%constant bool constbool=false;

%inline %{

// bool variables
bool bool1 = true;
bool bool2 = false;
bool* pbool = &bool1;
bool& rbool = bool2;
const bool* const_pbool = pbool;
const bool& const_rbool = rbool;

static int eax()
{
  return 1024;  // NOTE: any number > 255 should do
}

// bool functions
bool bo(bool b) {
  return b;
}
bool& rbo(bool& b) {
    return b;
}
bool* pbo(bool* b) {
    return b;
}
const bool& const_rbo(const bool& b) {
    return b;
}
const bool* const_pbo(const bool* b) {
    return b;
}

// helper function
bool value(bool* b) {
    return *b;
}

struct BoolStructure {
  bool m_bool1;
  bool m_bool2;
  bool* m_pbool;
  bool& m_rbool;
  const bool* m_const_pbool;
  const bool& m_const_rbool;
  BoolStructure() :
    m_bool1(true),
    m_bool2(false),
    m_pbool(&m_bool1),
    m_rbool(m_bool2),
    m_const_pbool(m_pbool),
    m_const_rbool(m_rbool) {}
private:
  BoolStructure& operator=(const BoolStructure &);
};
%}

