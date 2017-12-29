# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

# This file illustrates the proxy class C++ interface generated
# by SWIG.

swigexample

# ----- Object creation -----

printf("Creating some objects:\n");
c = swigexample.Circle(10)
s = swigexample.Square(10)

# ----- Access a static member -----

printf("\nA total of %i shapes were created\n", swigexample.Shape.nshapes);

# ----- Member data access -----

# Set the location of the object

c.x = 20
c.y = 30

s.x = -10
s.y = 5

printf("\nHere is their current position:\n");
printf("    Circle = (%f, %f)\n",c.x,c.y);
printf("    Square = (%f, %f)\n",s.x,s.y);

# ----- Call some methods -----

printf("\nHere are some properties of the shapes:\n");
function print_shape(o)
      o
      printf("  area      = %f\n", o.area());
      printf("  perimeter = %f\n", o.perimeter());
end;
print_shape(c);
print_shape(s);

printf("\nGuess I'll clean up now\n");

# Note: this invokes the virtual destructor
clear c
clear s

printf("%i shapes remain\n", swigexample.Shape.nshapes);
printf("Goodbye\n");
