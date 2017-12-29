require 'example'

# This should invoke foo(double)
Example.foo(3.14159)

# This should invoke foo(double, char *)
Example.foo(3.14159, "Pi")

# This should invoke foo(int, int)
Example.foo(3, 4)

# This should invoke foo(char *)
Example.foo("This is a test")

# This should invoke foo(long)
Example.foo(42)

# This should invoke Bar::Bar() followed by foo(Bar *)
Example.foo(Example::Bar.new)

# Skip a line
puts ""

# Each of the following three calls should invoke spam(int, int, int)
Example.spam(3)
Example.spam(3, 4)
Example.spam(3, 4, 5)

# Skip a line
puts ""

# Each of the following three calls should invoke spam(double, int, int)
Example.spam(3.0)
Example.spam(3.0, 4)
Example.spam(3.0, 4, 5)

# Skip a line
puts ""

# This should invoke Bar::Bar(double)
Example::Bar.new(3.14159)

# This should invoke Bar::Bar(double, char *)
Example::Bar.new(3.14159, "Pi")

# This should invoke Bar::Bar(int, int)
Example::Bar.new(3, 4)

# This should invoke Bar::Bar(char *)
Example::Bar.new("This is a test")

# This should invoke Bar::Bar(int)
Example::Bar.new(42)

# This should invoke Bar::Bar() for the input argument,
# followed by Bar::Bar(const Bar&).
Example::Bar.new(Example::Bar.new)

# Skip a line
puts ""

# Construct a new Bar instance (invokes Bar::Bar())
bar = Example::Bar.new

# This should invoke Bar::foo(double)
bar.foo(3.14159)

# This should invoke Bar::foo(double, char *)
bar.foo(3.14159, "Pi")

# This should invoke Bar::foo(int, int)
bar.foo(3, 4)

# This should invoke Bar::foo(char *)
bar.foo("This is a test")

# This should invoke Bar::foo(int)
bar.foo(42)

# This should invoke Bar::Bar() to construct the input
# argument, followed by Bar::foo(Bar *).
bar.foo(Example::Bar.new)

# This should invoke Bar::spam(int x, int y, int z)
bar.spam(1)

# This should invoke Bar::spam(double x, int y, int z)
bar.spam(3.14159)
