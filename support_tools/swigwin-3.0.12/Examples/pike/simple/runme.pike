int main()
{
    /* Call our gcd() function */
    int x = 42;
    int y = 105;
    int g = .example.gcd(x, y);
    write("The gcd of %d and %d is %d\n", x, y, g);

    /* Manipulate the Foo global variable */
    /* Output its current value */
    write("Foo = %f\n", .example->Foo_get());

    /* Change its value */
    .example->Foo_set(3.1415926);

    /* See if the change took effect */
    write("Foo = %f\n", .example->Foo_get());
    
    return 0;
}
