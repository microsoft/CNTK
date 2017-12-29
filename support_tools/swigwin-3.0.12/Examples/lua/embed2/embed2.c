/* embed2.c some more tests for an embedded interpreter
 
This will go a bit further as it will pass values to and from the lua code.
It uses less of the SWIG code, and more of the raw lua API's
 
What it will do is load the wrapped lib, load runme.lua and then call some functions.
To make life easier, all the printf's have either [C] or [Lua] at the start
so you can see where they are coming from.
 
We will be using the luaL_dostring()/lua_dostring() function to call into lua 
 
*/

/* Deal with Microsoft's attempt at deprecating C standard runtime functions */
#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif

/* Deal with Microsoft's attempt at deprecating methods in the standard C++ library */
#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
#include <stdarg.h>
#include <string.h>

#if LUA_VERSION_NUM > 501
#define lua_open luaL_newstate
#endif

/* the SWIG wrapped library */
extern int luaopen_example(lua_State*L);

/* This is an example of how to call the Lua function
    int add(int,int) 
  its very tedious, but gives you an idea of the issues involved.
  (look below for a better idea)
*/
int call_add(lua_State *L,int a,int b,int* res) {
  int top;
  /* ok, here we go:
  push a, push b, call 'add' check & return res
  */
  top=lua_gettop(L);  /* for later */
  lua_getglobal(L, "add");               /* function to be called */
  if (!lua_isfunction(L,-1)) {
    printf("[C] error: cannot find function 'add'\n");
    lua_settop(L,top);
    return 0;
  }
  lua_pushnumber(L,a);
  lua_pushnumber(L,b);
  if (lua_pcall(L, 2, 1, 0) != 0)  /* call function with 2 arguments and 1 result */
  {
    printf("[C] error running function `add': %s\n",lua_tostring(L, -1));
    lua_settop(L,top);
    return 0;
  }
  /* check results */
  if (!lua_isnumber(L,-1)) {
    printf("[C] error: returned value is not a number\n");
    lua_settop(L,top);
    return 0;
  }
  *res=(int)lua_tonumber(L,-1);
  lua_settop(L,top);  /* reset stack */
  return 1;
}

/* This is a variargs call function for calling from C into Lua.
Original Code from Programming in Lua (PIL) by Roberto Ierusalimschy
ISBN 85-903798-1-7 
http://www.lua.org/pil/25.3.html
This has been modified slightly to make it compile, and it's still a bit rough.
But it gives the idea of how to make it work.
*/
int call_va (lua_State *L,const char *func, const char *sig, ...) {
  va_list vl;
  int narg, nres;  /* number of arguments and results */
  int top;
  top=lua_gettop(L);  /* for later */

  va_start(vl, sig);
  lua_getglobal(L, func);  /* get function */

  /* push arguments */
  narg = 0;
  while (*sig) {  /* push arguments */
    switch (*sig++) {

    case 'd':  /* double argument */
      lua_pushnumber(L, va_arg(vl, double));
      break;

    case 'i':  /* int argument */
      lua_pushnumber(L, va_arg(vl, int));
      break;

    case 's':  /* string argument */
      lua_pushstring(L, va_arg(vl, char *));
      break;

    case '>':
      goto endwhile;

    default:
      printf("invalid option (%c)\n", *(sig - 1));
      goto fail;
    }
    narg++;
    /* do we need this?*/
    /* luaL_checkstack(L, 1, "too many arguments"); */
  }
endwhile:

  /* do the call */
  nres = (int)strlen(sig);  /* number of expected results */
  if (lua_pcall(L, narg, nres, 0) != 0)  /* do the call */
  {
    printf("error running function `%s': %s\n",func, lua_tostring(L, -1));
    goto fail;
  }

  /* retrieve results */
  nres = -nres;  /* stack index of first result */
  while (*sig) {  /* get results */
    switch (*sig++) {

    case 'd':  /* double result */
      if (!lua_isnumber(L, nres)) {
        printf("wrong result type\n");
        goto fail;
      }
      *va_arg(vl, double *) = lua_tonumber(L, nres);
      break;

    case 'i':  /* int result */
      if (!lua_isnumber(L, nres)) {
        printf("wrong result type\n");
        goto fail;
      }
      *va_arg(vl, int *) = (int)lua_tonumber(L, nres);
      break;

    case 's':  /* string result */
      if (!lua_isstring(L, nres)) {
        printf("wrong result type\n");
        goto fail;
      }
      strcpy(va_arg(vl, char *),lua_tostring(L, nres));/* WARNING possible buffer overflow */
      break;

    default: {
        printf("invalid option (%c)", *(sig - 1));
        goto fail;
      }
    }
    nres++;
  }
  va_end(vl);

  lua_settop(L,top);  /* reset stack */
  return 1; /* ok */
fail:
  lua_settop(L,top);  /* reset stack */
  return 0;   /* error */
}

int main(int argc,char* argv[]) {
  lua_State *L;
  int ok;
  int res;
  char str[80];
  printf("[C] Welcome to the simple embedded Lua example v2\n");
  printf("[C] We are in C\n");
  printf("[C] opening a Lua state & loading the libraries\n");
  L=lua_open();
  luaopen_base(L);
  luaopen_string(L);
  luaopen_math(L);
  printf("[C] now loading the SWIG wrappered library\n");
  luaopen_example(L);
  printf("[C] all looks ok\n");
  printf("\n");
  if (argc != 2 || argv[1] == NULL || strlen(argv[1]) == 0) {
    printf("[C] ERROR: no lua file given on command line\n");
    exit(3);
  }
  printf("[C] let's load the file '%s'\n", argv[1]);
  printf("[C] any lua code in this file will be executed\n");
  if (luaL_loadfile(L, argv[1]) || lua_pcall(L, 0, 0, 0)) {
    printf("[C] ERROR: cannot run lua file: %s",lua_tostring(L, -1));
    exit(3);
  }
  printf("[C] We are now back in C, all looks ok\n");
  printf("\n");
  printf("[C] let's call the Lua function 'add(1,1)'\n");
  printf("[C] using the C function 'call_add'\n");
  ok=call_add(L,1,1,&res);
  printf("[C] the function returned %d with value %d\n",ok,res);
  printf("\n");
  printf("[C] let's do this rather easier\n");
  printf("[C] we will call the same Lua function using a generic C function 'call_va'\n");
  ok=call_va(L,"add","ii>i",1,1,&res);
  printf("[C] the function returned %d with value %d\n",ok,res);
  printf("\n");
  printf("[C] we will now use the same generic C function to call 'append(\"cat\",\"dog\")'\n");
  ok=call_va(L,"append","ss>s","cat","dog",str);
  printf("[C] the function returned %d with value %s\n",ok,str);
  printf("\n");
  printf("[C] we can also make some bad calls to ensure the code doesn't fail\n");
  printf("[C] calling adds(1,2)\n");
  ok=call_va(L,"adds","ii>i",1,2,&res);
  printf("[C] the function returned %d with value %d\n",ok,res);
  printf("[C] calling add(1,'fred')\n");
  ok=call_va(L,"add","is>i",1,"fred",&res);
  printf("[C] the function returned %d with value %d\n",ok,res);
  printf("\n");
  printf("[C] Note: no protection if you mess up the va-args, this is C\n");
  printf("\n");
  printf("[C] Finally we will call the wrappered gcd function gdc(6,9):\n");
  printf("[C] This will pass the values to Lua, then call the wrappered function\n");
  printf("    Which will get the values from Lua, call the C code \n");
  printf("    and return the value to Lua and eventually back to C\n");
  printf("[C] Certainly not the best way to do it :-)\n");
  ok=call_va(L,"gcd","ii>i",6,9,&res);
  printf("[C] the function returned %d with value %d\n",ok,res);
  printf("\n");
  printf("[C] all finished, closing the lua state\n");
  lua_close(L);
  return 0;
}
