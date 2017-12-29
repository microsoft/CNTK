# file: runme.py

import sys
import example

# Call printf
example.printf("Hello World. I'm printf\n")

# Note: We call printf, but use *python* string formatting
for i in range(0, 10):
    example.printf("i is %d\n" % i)

# This will probably be garbled because %d is interpreted by C
example.printf("The value is %d\n")

stdout = example.stdout_stream()
# Call fprintf
example.fprintf(stdout, "Hello World. I'm fprintf\n")
for i in range(0, 10):
    example.fprintf(stdout, "i is %d\n" % i)

# This won't be garbled since %d is not interpreted
example.fprintf(stdout, "The value is %d\n")

# This function calls our NULL-terminated function

example.printv("Hello", "World", "this", "is", "a", "test.")
