using System;
using System.Threading;
using csharp_typemapsNamespace;

public class runme
{
  static void Main() 
  {
    // Test the C# types customisation by modifying the default char * typemaps to return a single char
    Things things = new Things();
    System.Text.StringBuilder initialLetters = new System.Text.StringBuilder();
    char myChar = things.start("boo");
    initialLetters.Append(myChar);
    myChar = Things.stop("hiss");
    initialLetters.Append(myChar);
    myChar = csharp_typemaps.partyon("off");
    initialLetters.Append(myChar);
    if (initialLetters.ToString() != "bho")
      throw new Exception("initial letters failed");

    // $csinput expansion
    csharp_typemaps.myInt = 1;
    try {
      csharp_typemaps.myInt = -1;
      throw new Exception("oops");
    } catch (ApplicationException) {
    }

    // Eager garbage collector test
    {
      const int NUM_THREADS = 8;
      Thread[] threads = new Thread[NUM_THREADS];
      TestThread[] testThreads = new TestThread[NUM_THREADS];
      // invoke the threads
      for (int i=0; i<NUM_THREADS; i++) {
        testThreads[i] = new TestThread(i);
        threads[i] = new Thread(new ThreadStart(testThreads[i].Run));
        threads[i].Start();
      }
      // wait for the threads to finish
      for (int i=0; i<NUM_THREADS; i++) {
        threads[i].Join();
      }
      for (int i=0; i<NUM_THREADS; i++) {
        if (testThreads[i].Failed) throw new Exception("Test Failed");
      }
    }

  }
}


public class TestThread {
   private int threadId;
   public bool Failed;
   public TestThread(int id) {
       threadId = id;
   }
   public void Run() {
     Failed = false;
     try {
       // Older versions of SWIG used IntPtr instead of HandleRef to hold the underlying
       // C++ pointer, so this test would (usually) fail as the garbage collector would
       // sometimes collect the Number class while it was being used in unmanaged code
       for (int i=0; i<5000; i++) { // run test for a few seconds
         {
           Obj obj = new Obj();
           Number n = new Number(i);
           Number triple = obj.triple(n);
           if (triple.Value != i*3)
             throw new ApplicationException("triple failed: " + triple.Value);
         }
         {
           Obj obj = new Obj();
           Number n = new Number(i);
           Number times6 = obj.times6(n);
           if (times6.Value != i*6)
             throw new ApplicationException("times6 failed: " + times6.Value);
         }
         {
           Obj obj = new Obj();
           Number n = new Number(i);
           Number times9 = obj.times9(n);
           if (times9.Value != i*9)
             throw new ApplicationException("times9 failed: " + times9.Value);
         }
         {
           Number n = new Number(i);
           Number quadruple = csharp_typemaps.quadruple(n);
           if (quadruple.Value != i*4)
             throw new ApplicationException("quadruple failed: " + quadruple.Value);
         }
         {
           Number n = new Number(i);
           Number times8 = csharp_typemaps.times8(n);
           if (times8.Value != i*8)
             throw new ApplicationException("times8 failed: " + times8.Value);
         }
         {
           Number n = new Number(i);
           Number times12 = csharp_typemaps.times12(n);
           if (times12.Value != i*12)
             throw new ApplicationException("times12 failed: " + times12.Value);
         }
       }
     } catch (Exception e) {
       Console.Error.WriteLine("Test failed (thread " + threadId + "): " + e.Message);
       Failed = true;
     }
   }
}


