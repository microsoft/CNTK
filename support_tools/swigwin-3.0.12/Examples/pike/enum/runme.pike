int main()
{
    write("*** color ***\n");
    write("    RED    = " + .example.RED + "\n");
    write("    BLUE   = " + .example.BLUE + "\n");
    write("    GREEN  = " + .example.GREEN + "\n");

    write("\n*** Foo::speed ***\n");
    write("    Foo_IMPULSE   = " + .example.Foo.IMPULSE + "\n");
    write("    Foo_WARP      = " + .example.Foo.WARP + "\n");
    write("    Foo_LUDICROUS = " + .example.Foo.LUDICROUS + "\n");

    write("\nTesting use of enums with functions\n\n");

    .example.enum_test(.example.RED,   .example.Foo.IMPULSE);
    .example.enum_test(.example.BLUE,  .example.Foo.WARP);
    .example.enum_test(.example.GREEN, .example.Foo.LUDICROUS);
    .example.enum_test(1234, 5678);

    write("\nTesting use of enum with class method\n");
    .example.Foo f = .example.Foo();

    f->enum_test(.example.Foo.IMPULSE);
    f->enum_test(.example.Foo.WARP);
    f->enum_test(.example.Foo.LUDICROUS);
    
    return 0;
}
