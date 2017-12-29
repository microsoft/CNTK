%module example

%{
#include "example.h"
%}

/* Tell SWIG that create_animal creates a new object */
%newobject Zoo::create_animal;

/* Keep track of mappings between C/C++ structs/classes
   and Ruby objects so we can implement a mark function. */
%trackobjects;


/* Specify the mark function */
%markfunc Zoo "mark_Zoo";

%include "example.h"

%header %{
	static void mark_Zoo(void* ptr) {
		Zoo* zoo = (Zoo*) ptr;

		/* Loop over each object and tell the garbage collector
		   that we are holding a reference to them. */
		int count = zoo->get_num_animals();

		for(int i = 0; i < count; ++i) {
			Animal* animal = zoo->get_animal(i);
			VALUE object = SWIG_RubyInstanceFor(animal);

			if (object != Qnil) {
				rb_gc_mark(object);
			}
		}
	}
%}
