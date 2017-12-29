
using System;
using System.Reflection;

public class runme {

  public static void Main(String[] args) {


    int a = 37;
    int b = 42;

    // Now call our C function with a bunch of callbacks

    Console.WriteLine( "Trying some C callback functions" );
    Console.WriteLine( "    a        = " + a );
    Console.WriteLine( "    b        = " + b );
    Console.WriteLine( "    ADD(a,b) = " + example.do_op(a,b,example.ADD) );
    Console.WriteLine( "    SUB(a,b) = " + example.do_op(a,b,example.SUB) );
    Console.WriteLine( "    MUL(a,b) = " + example.do_op(a,b,example.MUL) );

    Console.WriteLine( "Here is what the C callback function classes are called in C#" );
    Console.WriteLine( "    ADD      = " + example.ADD.GetType() );
    Console.WriteLine( "    SUB      = " + example.SUB.GetType() );
    Console.WriteLine( "    MUL      = " + example.MUL.GetType() );
  }
}
