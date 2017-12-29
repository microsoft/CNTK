/* File : example.i */
%module example

%{
#include "example.h"
%}

// before we grab the header file, we must warn SWIG about some of these functions.

// these functions create data, so must be managed
%newobject createCircle;
%newobject createSquare;

// this method returns as pointer which must be managed
%newobject ShapeOwner::remove;

// you cannot use %delobject on ShapeOwner::add()
// as this disowns the ShapeOwner, not the Shape (oops)
//%delobject ShapeOwner::add(Shape*); DO NOT USE

// either you can use a new function (such as this)
/*%delobject add_Shape;
%inline %{
void add_Shape(Shape* s,ShapeOwner* own){own->add(s);}
%}*/

// or a better solution is a typemap
%apply SWIGTYPE *DISOWN {Shape* ptr};

// now we can grab the header file
%include "example.h"

