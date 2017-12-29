// This example illustrates global variable access from C#.

using System;
using System.Reflection;

public class runme {

  public static void Main() {

    // Try to set the values of some global variables

    example.ivar   =  42;
    example.svar   = -31000;
    example.lvar   =  65537;
    example.uivar  =  123456;
    example.usvar  =  61000;
    example.ulvar  =  654321;
    example.scvar  =  -13;
    example.ucvar  =  251;
    example.cvar   =  'S';
    example.fvar   =  (float)3.14159;
    example.dvar   =  2.1828;
    example.strvar =  "Hello World";
    example.iptrvar= example.new_int(37);
    example.ptptr  = example.new_Point(37,42);
    example.name   = "Bill";

    // Now print out the values of the variables

    Console.WriteLine( "Variables (values printed from C#)" );

    Console.WriteLine( "ivar      =" + example.ivar );
    Console.WriteLine( "svar      =" + example.svar );
    Console.WriteLine( "lvar      =" + example.lvar );
    Console.WriteLine( "uivar     =" + example.uivar );
    Console.WriteLine( "usvar     =" + example.usvar );
    Console.WriteLine( "ulvar     =" + example.ulvar );
    Console.WriteLine( "scvar     =" + example.scvar );
    Console.WriteLine( "ucvar     =" + example.ucvar );
    Console.WriteLine( "fvar      =" + example.fvar );
    Console.WriteLine( "dvar      =" + example.dvar );
    Console.WriteLine( "cvar      =" + example.cvar );
    Console.WriteLine( "strvar    =" + example.strvar );
    Console.WriteLine( "cstrvar   =" + example.cstrvar );
    Console.WriteLine( "iptrvar   =" + example.iptrvar );
    Console.WriteLine( "name      =" + example.name );
    Console.WriteLine( "ptptr     =" + example.ptptr + example.Point_print(example.ptptr) );
    Console.WriteLine( "pt        =" + example.pt + example.Point_print(example.pt) );

    Console.WriteLine( "\nVariables (values printed from C)" );

    example.print_vars();

    Console.WriteLine( "\nNow I'm going to try and modify some read only variables" );
    Console.WriteLine( "\nChecking that the read only variables are readonly..." );

    example ex = new example();
    Type t = ex.GetType();

    Console.WriteLine( "     'path'" );
    PropertyInfo pi = t.GetProperty("path");
    if (pi.CanWrite)
      Console.WriteLine("Oh dear this variable is not read only\n");
    else
      Console.WriteLine("Good.");

    Console.WriteLine( "     'status'" );
    pi = t.GetProperty("status");
    if (pi.CanWrite)
      Console.WriteLine("Oh dear this variable is not read only");
    else
      Console.WriteLine("Good.");

    Console.WriteLine( "\nI'm going to try and update a structure variable.\n" );

    example.pt = example.ptptr;

    Console.WriteLine( "The new value is" );
    example.pt_print();
    Console.WriteLine( "You should see the value" + example.Point_print(example.ptptr) );
  }
}
