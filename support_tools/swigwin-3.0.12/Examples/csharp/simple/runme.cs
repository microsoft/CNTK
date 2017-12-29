using System;

public class runme
{
    static void Main() 
    {
        // Call our gcd() function

        int x = 42;
        int y = 105;
        int g = example.gcd(x,y);
        Console.WriteLine("The gcd of " + x + " and " + y + " is " + g);

        // Manipulate the Foo global variable

        // Output its current value
        Console.WriteLine("Foo = " + example.Foo);

        // Change its value
        example.Foo = 3.1415926;

        // See if the change took effect
        Console.WriteLine("Foo = " + example.Foo);
    }
}
