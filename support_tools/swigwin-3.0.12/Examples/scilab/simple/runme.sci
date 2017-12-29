lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// Call our gcd() function

x = 42;
y = 105;
g = gcd(x,y);
printf("The gcd of %d and %d is %d\n",x,y,g);

// Manipulate the Foo global variable

// Get its default value (see in example.c)
defaultValue = Foo_get()
if defaultValue <> 3 then pause; end

// Change its value
Foo_set(3.1415926)

// See if the change took effect
if Foo_get() <> 3.1415926 then pause,end

exit

