# file: runme.rb

require 'example'

# Call our gcd() function

x = 42
y = 105
g = Example.gcd(x,y)
printf "The gcd of %d and %d is %d\n",x,y,g

# Call the gcdmain() function
Example.gcdmain(["gcdmain","42","105"])

# Call the count function

printf "%d\n",Example.count("Hello World","l")

# Call the capitalize function

printf "%s\n",Example.capitalize("hello world")

