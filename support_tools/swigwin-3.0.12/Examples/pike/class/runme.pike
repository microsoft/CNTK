import .example;

int main()
{
    // ----- Object creation -----

    write("Creating some objects:\n");
    Circle c = Circle(10.0);
    write("    Created circle.\n");
    Square s = Square(10.0);
    write("    Created square.\n");

    // ----- Access a static member -----

    write("\nA total of " + Shape_nshapes_get() + " shapes were created\n");

    // ----- Member data access -----

    // Set the location of the object

    c->x_set(20.0);
    c->y_set(30.0);

    s->x_set(-10.0);
    s->y_set(5.0);

    write("\nHere is their current position:\n");
    write("    Circle = (%f, %f)\n", c->x_get(), c->y_get());
    write("    Square = (%f, %f)\n", s->x_get(), s->y_get());

    // ----- Call some methods -----

    write("\nHere are some properties of the shapes:\n");
    write("   The circle:\n");
    write("        area      = %f.\n", c->area());
    write("        perimeter = %f.\n", c->perimeter());
    write("   The square:\n");
    write("        area      = %f.\n", s->area());
    write("        perimeter = %f.\n", s->perimeter());

    write("\nGuess I'll clean up now\n");

    /* See if we can force 's' to be garbage-collected */
    s = 0;
    
    /* Now we should be down to only 1 shape */
    write("%d shapes remain\n", Shape_nshapes_get());
    
    /* Done */
    write("Goodbye\n");
    
    return 0;
}
