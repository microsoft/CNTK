/* File : example.i */
/*
This demonstrates how to pass a lua function, into some C code and then call it.

There are two examples, the first is as a parameter, the second as a global variable.

*/
%module example
%{
#include "example.h"
%}
/* the extra wrappers for lua functions, see SWIG/Lib/lua/lua_fnptr.i for more details */
%include "lua_fnptr.i"

/* these are a bunch of C functions which we want to be able to call from lua */
extern int add(int,int);
extern int sub(int,int);
extern int mul(int,int);

/* this function takes a lua function as a parameter and calls it.
As this is takes a lua fn it needs lua code
*/
%inline %{
	
int callback(int a, int b, SWIGLUA_FN fn)
{
	SWIGLUA_FN_GET(fn);
	lua_pushnumber(fn.L,a);
	lua_pushnumber(fn.L,b);
	lua_call(fn.L,2,1);    /* 2 in, 1 out */
	return (int)luaL_checknumber(fn.L,-1);
}	
%}	

/******************
Second code uses a stored reference.
*******************/

%inline %{
/* note: this is not so good to just have it as a raw ref
 people could set anything to this
 a better solution would to be to have a fn which wants a SWIGLUA_FN, then
 checks the type & converts to a SWIGLUA_REF.
*/	
SWIGLUA_REF the_func={0,0};
	
void call_the_func(int a)
{
	int i;
	if (the_func.L==0){
		printf("the_func is zero\n");
		return;
	}
	swiglua_ref_get(&the_func);
	if (!lua_isfunction(the_func.L,-1))
	{
		printf("the_func is not a function\n");
		return;
	}
	lua_pop(the_func.L,1); /* tidy stack */
	for(i=0;i<a;i++)
	{
		swiglua_ref_get(&the_func);
		lua_pushnumber(the_func.L,i);
		lua_call(the_func.L,1,0);    /* 1 in, 0 out */
	}
}	

%}
