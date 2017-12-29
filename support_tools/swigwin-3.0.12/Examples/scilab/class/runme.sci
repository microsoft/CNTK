lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// ----- Object creation -----

printf("Creating some objects:\n");
c = new_Circle(10)
s = new_Square(10)

// ----- Access a static member -----

printf("\nA total of %i shapes were created\n", Shape_nshapes_get());

// ----- Member data access -----

// Set the location of the object

Shape_x_set(c, 20);
Shape_y_set(c, 30);

Shape_x_set(s, -10);
Shape_y_set(s, 5);

printf("\nHere is their current position:\n");
printf("    Circle = (%f, %f)\n", Shape_x_get(c), Shape_y_get(c));
printf("    Square = (%f, %f)\n", Shape_x_get(s), Shape_y_get(s));

// ----- Call some methods -----

printf("\nHere are some properties of the shapes:\n");
function print_shape(o)
      printf("  area      = %f\n", Shape_area(o));
      printf("  perimeter = %f\n", Shape_perimeter(o));
endfunction
print_shape(c);
print_shape(s);

printf("\nGuess I will clean up now\n");

// Note: this invokes the virtual destructor
delete_Circle(c);
delete_Square(s);

printf("%i shapes remain\n", Shape_nshapes_get());
printf("Goodbye\n");

exit
