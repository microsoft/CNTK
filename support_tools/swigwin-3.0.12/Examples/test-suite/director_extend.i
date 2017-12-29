
%module(directors="1") director_extend

%extend SpObject 
{
   virtual int dummy()          // Had to remove virtual to work
   {
      return $self->getFooBar();
   }
};

%inline %{
#ifndef SWIG_DIRECTORS
// dummy definition for non-director languages
namespace Swig {
  typedef int Director;
}
#endif
%}

// Some director implementations do not have Swig::director
#if !defined(SWIGGO)
%extend SpObject
{
  size_t ExceptionMethod()
  {
// Check positioning of director code in wrapper file
// Below is what we really want to test, but director exceptions vary too much across across all languages
//    throw Swig::DirectorException("DirectorException was not in scope!!");
// Instead check definition of Director class as that is defined in the same place as DirectorException (director.swg)
    size_t size = sizeof(Swig::Director);
    return size;
  }
}
#endif


%inline %{
class SpObject
{
public:
   SpObject() {}
   virtual ~SpObject() {}

   int getFooBar() const {
      return 666;
   }

private:
   // Do NOT define the assignment operator
   SpObject& operator=(const SpObject& rhs);

   // This class can not be copied.  Do NOT define the copy Constructor.
   SpObject (const SpObject& rhs);
};
%}

