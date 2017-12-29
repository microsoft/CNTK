// This example illustrates how C++ classes can be used from C# using SWIG.
// The C# class gets mapped onto the C++ class and behaves as if it is a C# class.

using System;

public class runme
{
    static void Main() 
    {
        // ----- Object creation -----

        Console.WriteLine( "Creating some objects:" );

        using (Square s = new Square(10))
        using (Circle c = new Circle(10))
        {
            Console.WriteLine( "    Created circle " + c );
            Console.WriteLine( "    Created square " + s );

            // ----- Access a static member -----

            Console.WriteLine( "\nA total of " + Shape.nshapes + " shapes were created" );

            // ----- Member data access -----

            // Notice how we can do this using functions specific to
            // the 'Circle' class.
            c.x = 20;
            c.y = 30;

            // Now use the same functions in the base class
            Shape shape = s;
            shape.x = -10;
            shape.y = 5;

            Console.WriteLine( "\nHere is their current position:" );
            Console.WriteLine( "    Circle = (" + c.x + " " + c.y + ")" );
            Console.WriteLine( "    Square = (" + s.x + " " + s.y + ")" );

            // ----- Call some methods -----

            Console.WriteLine( "\nHere are some properties of the shapes:" );
            Shape[] shapes = {c,s};
            //            for (int i=0; i<shapes.Size; i++)
            for (int i=0; i<2; i++)
            {
                Console.WriteLine( "   " + shapes[i].ToString() );
                Console.WriteLine( "        area      = " + shapes[i].area() );
                Console.WriteLine( "        perimeter = " + shapes[i].perimeter() );
            }

            // Notice how the area() and perimeter() functions really
            // invoke the appropriate virtual method on each object.

            // ----- Delete everything -----

            Console.WriteLine( "\nGuess I'll clean up now" );

        }
        // Note: when this using scope is exited the C# Dispose() methods 
        // are called which in turn call the C++ destructors

        Console.WriteLine( Shape.nshapes + " shapes remain" );
        Console.WriteLine( "Goodbye" );
    }
}
