# file: runme.R

dyn.load(paste("example", .Platform$dynlib.ext, sep=""))
source("example.R")
cacheMetaData(1)

# Call our gcd() function

x <- 42
y <- 105
g <- gcd(x,y)
sprintf("The gcd of %d and %d is %d", x, y, g)

# Manipulate the Foo global variable

# Output its current value
Foo()

# Change its value
Foo(3.1415926)

# See if the change took effect
Foo()

