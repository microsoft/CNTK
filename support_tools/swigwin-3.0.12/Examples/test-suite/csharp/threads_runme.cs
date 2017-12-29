using System;
using System.Threading;
using threadsNamespace;

public class runme
{
    static void Main() 
    {
      Kerfuffle kerf = new Kerfuffle();
      const int NUM_THREADS = 8;
      Thread[] threads = new Thread[NUM_THREADS];
      TestThread[] testThreads = new TestThread[NUM_THREADS];
      // invoke the threads
      for (int i=0; i<NUM_THREADS; i++) {
        testThreads[i] = new TestThread(kerf, i);
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

public class TestThread {
   private int threadId;
   private Kerfuffle kerf;
   public bool Failed;
   public TestThread(Kerfuffle t, int id) {
       kerf = t;
       threadId = id;
   }
   public void Run() {
     Failed = false;
     try {
       for (int i=0; i<30000; i++) { // run test for a few seconds on a 1GHz machine
         string given = "This is the test string that should come back. A number: " + i;
         string received = kerf.StdString(given);
         if (received != given) {
           throw new ApplicationException("StdString string does not match. Received:\n[" + received + "].\nExpected:\n{" + given + "}");
         }
       }
       for (int i=0; i<30000; i++) { // run test for a few seconds on a 1GHz machine
         string given = "This is the test string that should come back. A number: " + i;
         string received = kerf.CharString(given);
         if (received != given) {
           throw new ApplicationException("StdString string does not match. Received:\n[" + received + "].\nExpected:\n{" + given + "}");
         }
       }
     } catch (Exception e) {
       Console.Error.WriteLine("Test failed (thread " + threadId + "): " + e.Message);
       Failed = true;
     }
   }
}


