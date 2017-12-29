// This example illustrates how C++ templates can be used from C#.

using System;

public class runme {

  public static void Main() 
  {
    // Call some templated functions
    Console.WriteLine(example.maxint(3,7));
    Console.WriteLine(example.maxdouble(3.14,2.18));
    
    // Create some class
    
    vecint iv = new vecint(100);
    vecdouble dv = new vecdouble(1000);
    
    for (int i=0; i<100; i++)
        iv.setitem(i,2*i);
    
    for (int i=0; i<1000; i++)
          dv.setitem(i, 1.0/(i+1));
    
    {
        int sum = 0;
        for (int i=0; i<100; i++)
            sum = sum + iv.getitem(i);

        Console.WriteLine(sum);
    }
    
    {
        double sum = 0.0;
        for (int i=0; i<1000; i++)
            sum = sum + dv.getitem(i);
        Console.WriteLine(sum);
    }
  }
}
