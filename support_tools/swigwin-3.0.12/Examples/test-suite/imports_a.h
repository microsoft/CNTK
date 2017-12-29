#ifndef a_h
#define a_h
enum GlobalEnum { globalenum1=1, globalenum2 };

/* This function should be static as it will be emitted into the code for
 * every module.  All _static targets would fail here with a multiple 
 * definition if this is not static. */
static GlobalEnum global_test(GlobalEnum e) { return e; }

class A { 
 public: 
  A() {}
  virtual ~A() {}
  
  void hello() {}

  enum MemberEnum { memberenum1=10, memberenum2 };
  virtual MemberEnum member_virtual_test(MemberEnum e) { return e; }
  virtual GlobalEnum global_virtual_test(GlobalEnum e) { return global_test(e); }
};

/* This class overrides nothing. Inherited classes should see base functions.
*/
class A_Intermediate : public A { 
 public:
  A_Intermediate(){}
  ~A_Intermediate(){}
};
#endif
