// import .example;

int main()
{
	// This should invoke foo(double)
	.example.foo(3.14159);

	// This should invoke foo(double, char *)
	.example.foo(3.14159, "Pi");

	// This should invoke foo(int, int)
	.example.foo(3, 4);

	//  This should invoke foo(char *)
	.example.foo("This is a test");

	// This should invoke foo(long)
	.example.foo(42);

	/*
	// This should invoke Bar::Bar() followed by foo(Bar *)
	foo(Bar.new);

	// Skip a line
	write("\n");

	// This should invoke Bar::Bar(double)
	Bar.new(3.14159);

	// This should invoke Bar::Bar(double, char *)
	Bar.new(3.14159, "Pi");

	// This should invoke Bar::Bar(int, int)
	Bar.new(3, 4);

	// This should invoke Bar::Bar(char *)
	Bar.new("This is a test");

	// This should invoke Bar::Bar(int)
	Bar.new(42);

	// This should invoke Bar::Bar() for the input argument,
	// followed by Bar::Bar(const Bar&).
	Bar.new(Bar.new);

	// Skip a line
	write("\n");
	*/

	// Construct a new Bar instance (invokes Bar::Bar())
	/*
	bar = Bar.new;

	// This should invoke Bar::foo(double)
	bar.foo(3.14159);

	// This should invoke Bar::foo(double, char *)
	bar.foo(3.14159, "Pi");

	// This should invoke Bar::foo(int, int)
	bar.foo(3, 4);

	// This should invoke Bar::foo(char *)
	bar.foo("This is a test");

	// This should invoke Bar::foo(int)
	bar.foo(42);

	// This should invoke Bar::Bar() to construct the input
	// argument, followed by Bar::foo(Bar *).
	bar.foo(Example::Bar.new);

	// This should invoke Bar::spam(int x, int y, int z)
	bar.spam(1);

	// This should invoke Bar::spam(double x, int y, int z)
	bar.spam(3.14159);
	*/

   	write("Goodbye\n");
    
   	return 0;
}
