using System;
using li_std_auto_ptrNamespace;

public class li_std_auto_ptr_runme {
    private static void WaitForGC()
    {
        System.GC.Collect(); 
        System.GC.WaitForPendingFinalizers();
        System.Threading.Thread.Sleep(10);
    }

    public static void Main()
    {
        Klass k1 = li_std_auto_ptr.makeKlassAutoPtr("first");
        if (k1.getLabel() != "first")
            throw new Exception("wrong object label");

        Klass k2 = li_std_auto_ptr.makeKlassAutoPtr("second");
        if (Klass.getTotal_count() != 2)
            throw new Exception("number of objects should be 2");

        k1 = null;
        {
          int countdown = 500;
          int expectedCount = 1;
          while (true) {
            WaitForGC();
            if (--countdown == 0)
              break;
            if (Klass.getTotal_count() == expectedCount)
              break;
          };
          int actualCount = Klass.getTotal_count();
          if (actualCount != expectedCount)
            Console.Error.WriteLine("Expected count: " + expectedCount + " Actual count: " + actualCount); // Finalizers are not guaranteed to be run and sometimes they just don't
        }

        if (k2.getLabel() != "second")
            throw new Exception("wrong object label");

        k2 = null;
        {
          int countdown = 500;
          int expectedCount = 0;
          while (true) {
            WaitForGC();
            if (--countdown == 0)
              break;
            if (Klass.getTotal_count() == expectedCount)
              break;
          }
          int actualCount = Klass.getTotal_count();
          if (actualCount != expectedCount)
            Console.Error.WriteLine("Expected count: " + expectedCount + " Actual count: " + actualCount); // Finalizers are not guaranteed to be run and sometimes they just don't
        }
    }
}
