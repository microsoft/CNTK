#include <chicken/chicken_ext_test_wrap_hdr.h>
#include <imports_a.h>

void test_create(C_word,C_word,C_word) C_noret;
void test_create(C_word argc, C_word closure, C_word continuation) {
  C_word resultobj;
  swig_type_info *type;
  A *newobj;
  C_word *known_space = C_alloc(C_SIZEOF_SWIG_POINTER);

  C_trace("test-create");
  if (argc!=2) C_bad_argc(argc,2);


  newobj = new A();

  type = SWIG_TypeQuery("A *");
  resultobj = SWIG_NewPointerObj(newobj, type, 1);

  C_kontinue(continuation, resultobj);
}
