// This example illustrates how C++ classes can be used from Java using SWIG.
// The Java class gets mapped onto the C++ class and behaves as if it is a Java class.

public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // ----- Object creation -----
    
    System.out.println( "Creating some objects:" );
    Circle c = new Circle(10);
    System.out.println( "    Created circle " + c );
    Square s = new Square(10);
    System.out.println( "    Created square " + s );
    
    // ----- Access a static member -----
    
    System.out.println( "\nA total of " + Shape.getNshapes() + " shapes were created" );
    
    // ----- Member data access -----
    
    // Notice how we can do this using functions specific to
    // the 'Circle' class.
    c.setX(20);
    c.setY(30);
    
    // Now use the same functions in the base class
    Shape shape = s;
    shape.setX(-10);
    shape.setY(5);
    
    System.out.println( "\nHere is their current position:" );
    System.out.println( "    Circle = (" + c.getX() + " " + c.getY() + ")" );
    System.out.println( "    Square = (" + s.getX() + " " + s.getY() + ")" );
    
    // ----- Call some methods -----
    
    System.out.println( "\nHere are some properties of the shapes:" );
    Shape[] shapes = {c,s};
    for (int i=0; i<shapes.length; i++)
    {
          System.out.println( "   " + shapes[i].toString() );
          System.out.println( "        area      = " + shapes[i].area() );
          System.out.println( "        perimeter = " + shapes[i].perimeter() );
    }
    
    // Notice how the area() and perimeter() functions really
    // invoke the appropriate virtual method on each object.
    
    // ----- Delete everything -----
    
    System.out.println( "\nGuess I'll clean up now" );
    
    // Note: this invokes the virtual destructor
    // You could leave this to the garbage collector
    c.delete();
    s.delete();
    
    System.out.println( Shape.getNshapes() + " shapes remain" );
    System.out.println( "Goodbye" );
  }
}
