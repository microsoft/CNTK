/* 
 * This interface file tests whether the language modules handle the kind when declared 
 * with the function/member name, especially when used with proxy classes.
*/

%module kind

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) foo;  /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) bar;  /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) uni;  /* Ruby, wrong class name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) test; /* Ruby, wrong class name */

%inline %{

class foo {};
struct bar {};
union uni {};

struct test {
  void foofn(class foo myfoo1, foo myfoo2, class foo* myfoo3, foo* myfoo4, class foo& myfoo5, foo& myfoo6) {}
  void barfn(struct bar mybar1, bar mybar2, struct bar* mybar3, bar* mybar4, struct bar& mybar5, bar& mybar6) {}
  void unifn(union uni myuni1, uni myuni2, union uni* myuni3, uni* myuni4, union myuni& myuni5, myuni& myuni6) {}

  class foo myFooMember;
  struct bar myBarMember;
  union uni myUniMember;

  class foo* mypFooMember;
  struct bar* mypBarMember;
  union uni* mypUniMember;
};

%}

