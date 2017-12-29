int main()
{
    // Call some templated functions
    write(sprintf("%d\n", .example.maxint(3, 7)));
    write(sprintf("%f\n", .example.maxdouble(3.14, 2.18)));
    
    // Create some objects
    .example.vecint iv = .example.vecint(100);
    .example.vecdouble dv = .example.vecdouble(1000);
    
    for (int i = 0; i < 100; i++) {
        iv->setitem(i, 2*i);
    }

    for (int i = 0; i < 1000; i++) {
        dv->setitem(i, 1.0/(i+1));
    }
    
    int isum = 0;
    for (int i = 0; i < 100; i++) {
        isum += iv->getitem(i);
    }
    
    write(sprintf("%d\n", isum));
    
    float fsum = 0.0;
    for (int i = 0; i < 1000; i++) {
        fsum += dv->getitem(i);
    }
    write(sprintf("%f\n", fsum));
    
    return 0;
}
