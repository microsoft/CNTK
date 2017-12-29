using System;

public class runme
{
    static void Main() 
    {
        // Print out the value of some enums
        Console.WriteLine("*** color ***");
        Console.WriteLine("    " + color.RED + " = " + (int)color.RED);
        Console.WriteLine("    " + color.BLUE + " = " + (int)color.BLUE);
        Console.WriteLine("    " + color.GREEN + " = " + (int)color.GREEN);

        Console.WriteLine("\n*** Foo::speed ***");
        Console.WriteLine("    Foo::" + Foo.speed.IMPULSE + " = " + (int)Foo.speed.IMPULSE);
        Console.WriteLine("    Foo::" + Foo.speed.WARP + " = " + (int)Foo.speed.WARP);
        Console.WriteLine("    Foo::" + Foo.speed.LUDICROUS + " = " + (int)Foo.speed.LUDICROUS);

        Console.WriteLine("\nTesting use of enums with functions\n");

        example.enum_test(color.RED, Foo.speed.IMPULSE);
        example.enum_test(color.BLUE, Foo.speed.WARP);
        example.enum_test(color.GREEN, Foo.speed.LUDICROUS);

        Console.WriteLine( "\nTesting use of enum with class method" );
        Foo f = new Foo();

        f.enum_test(Foo.speed.IMPULSE);
        f.enum_test(Foo.speed.WARP);
        f.enum_test(Foo.speed.LUDICROUS);
    }
}
