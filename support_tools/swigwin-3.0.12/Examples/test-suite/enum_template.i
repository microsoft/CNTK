%module enum_template

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) eTest0;        /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) eTest1;        /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) eTest2;        /* Ruby, wrong class name */

/*
From bug report 992329:

In Python I see

>>> import enum_template
>>> enum_template.MakeETest()
'_60561408_p_ETest'
>>> enum_template.TakeETest(0)
Traceback (most recent call last):
File "<stdin>", line 1, in ?
TypeError: Expected a pointer

Without the %template, things work fine: the first
function call returns an integer, and the second
succeeds.
*/

%inline %{

enum ETest
{
eTest0,
eTest1,
eTest2
};

void TakeETest(ETest test) {}
ETest MakeETest(void) {return eTest1;}

template<class T> class CTempl
{
};

%}

%template(CTempl_ETest) CTempl<ETest>;
