/*
dual.cpp a test for multiple modules and multiple interpreters statically linked together.

Earlier version of lua bindings for SWIG would fail if statically linked.

What is happening is as follows:
example.i declares a type Foo
examples2.i declares Bar

The first lua state will load example.i 
and check to see if types Foo and Bar are registered with it
(Foo should be & Bar should not)

The second lua state will load example2.i
and check to see if types Foo and Bar are registered with it
(Bar should be & Foo should not)

Note: Though both the modules exist and are loaded, they are not linked together,
as they are connected to seperate lua interpreters.

When the third lua state loads both example.i and example2.i,
the two modules are now linked together, and all can now find
both Foo and Bar.
*/

#include "swigluarun.h"	// the swig runtimes

#include <stdio.h>
#include <stdlib.h>

// the 2 libraries which are wrapped via SWIG
extern "C" int luaopen_example(lua_State*L);
extern "C" int luaopen_example2(lua_State*L);

#define DEBUG(X) {printf(X);fflush(stdout);}
#define DEBUG2(X,Y) {printf(X,Y);fflush(stdout);}
#define DEBUG3(X,Y,Z) {printf(X,Y,Z);fflush(stdout);}

#if LUA_VERSION_NUM > 501
#define lua_open luaL_newstate
#endif

void testModule(lua_State *L)
{
  swig_type_info *pTypeInfo=0,*pTypeInfo2=0;
  swig_module_info *pModule=0;
  pModule=SWIG_GetModule(L);
  DEBUG2("  SWIG_GetModule() returns %p\n", (void *)pModule)
  if(pModule==0) return;
  pTypeInfo = SWIG_TypeQuery(L,"Foo *");
  DEBUG2("  Type (Foo*) is %s\n",pTypeInfo==0?"unknown":"known");
  DEBUG3("    Module %p typeinfo(Foo*) %p\n", (void *)pModule, (void *)pTypeInfo);
  pTypeInfo2 = SWIG_TypeQuery(L,"Bar *");
  DEBUG2("  Type (Bar*) is %s\n",pTypeInfo2==0?"unknown":"known");
  DEBUG3("    Module %p typeinfo(Bar*) %p\n", (void *)pModule, (void *)pTypeInfo2);
}

int main(int argc,char* argv[])
{
  lua_State *L1=0,*L2=0,*L3=0;

  printf("This is a test of having two SWIG'ed modules and three lua states\n"
	"statically linked together.\n"
	"Its mainly to check that all the types are correctly managed\n\n");
	
  DEBUG("creating lua states(L1,L2,L3)");
  L1=lua_open();
  L2=lua_open();
  L3=lua_open();
  DEBUG("ok\n\n");

  DEBUG("luaopen_example(L1)..");
  luaopen_example(L1);
  DEBUG("ok\n");
	
  DEBUG("Testing Module L1\n");
  DEBUG("This module should know about Foo* but not Bar*\n");
  testModule(L1);
  DEBUG("End Testing Module L1\n\n");

  DEBUG("luaopen_example2(L2)..");
  luaopen_example2(L2);
  DEBUG("ok\n");
	
  DEBUG("Testing Module L2\n");
  DEBUG("This module should know about Bar* but not Foo*\n");
  testModule(L2);
  DEBUG("End Testing Module L2\n\n");

  DEBUG("luaopen_example(L3)..");
  luaopen_example(L3);
  DEBUG("ok\n");
  DEBUG("luaopen_example2(L3)..");
  luaopen_example2(L3);
  DEBUG("ok\n");

  DEBUG("Testing Module L3\n");
  DEBUG("This module should know about Foo* and Bar*\n");
  testModule(L3);
  DEBUG("End Testing Module L3\n\n");

  DEBUG("Testing Module L1 again\n");
  DEBUG("It now should know about Foo* and Bar*\n");
  testModule(L1);
  DEBUG("End Testing Module L1 again\n\n");

  DEBUG("close all..");
  lua_close(L1);
  lua_close(L2);
  lua_close(L3);
  DEBUG("ok, exiting\n");
  return 0;
}
