%module cpp11_rvalue_reference3

%warnfilter(SWIGWARN_TYPEMAP_SWIGTYPELEAK);

%inline %{
#include <utility>
struct Thing {};

Thing && global_rvalue_ref = Thing();
Thing *&& global_rvalue_ref_ptr1 = 0;
Thing const*&& global_rvalue_ref_ptr2 = 0;
Thing *const&& global_rvalue_ref_ptr3 = 0;
Thing const*const &&global_rvalue_ref_ptr4 = 0;

Thing && returnit1() { return std::move(global_rvalue_ref); }
Thing *&& returnit2() { return std::move(global_rvalue_ref_ptr1); }
Thing const*&& returnit3() { return std::move(global_rvalue_ref_ptr2); }
Thing *const&& returnit4() { return std::move(global_rvalue_ref_ptr3); }
Thing const*const&& returnit5() { return std::move(global_rvalue_ref_ptr4); }

void takeit1(Thing && t) {}
void takeit2(Thing *&& t) {}
void takeit3(Thing const*&& t) {}
void takeit4(Thing *const&& t) {}
void takeit5(Thing const*const&& t) {}

struct Containing {
  Thing && member_rvalue_ref;
  Thing *&& member_rvalue_ref_ptr1 = 0;
  Thing const*&& member_rvalue_ref_ptr2 = 0;
  Thing *const&& member_rvalue_ref_ptr3 = 0;
  Thing const*const &&member_rvalue_ref_ptr4 = 0;

  Containing() : member_rvalue_ref(Thing()) {}
};
%}


%inline %{
int && int_global_rvalue_ref = 5;
int *&& int_global_rvalue_ref_ptr1 = 0;
int const*&& int_global_rvalue_ref_ptr2 = 0;
int *const&& int_global_rvalue_ref_ptr3 = 0;
int const*const &&int_global_rvalue_ref_ptr4 = 0;

int && int_returnit1() { return std::move(int_global_rvalue_ref); }
int *&& int_returnit2() { return std::move(int_global_rvalue_ref_ptr1); }
int const*&& int_returnit3() { return std::move(int_global_rvalue_ref_ptr2); }
int *const&& int_returnit4() { return std::move(int_global_rvalue_ref_ptr3); }
int const*const&& int_returnit5() { return std::move(int_global_rvalue_ref_ptr4); }

void int_takeit1(int && t) {}
void int_takeit2(int *&& t) {}
void int_takeit3(int const*&& t) {}
void int_takeit4(int *const&& t) {}
void int_takeit5(int const*const&& t) {}

struct IntContaining {
  int && member_rvalue_ref;
  int *&& member_rvalue_ref_ptr1 = 0;
  int const*&& member_rvalue_ref_ptr2 = 0;
  int *const&& member_rvalue_ref_ptr3 = 0;
  int const*const &&member_rvalue_ref_ptr4 = 0;

  IntContaining() : member_rvalue_ref(55) {}
};
%}
