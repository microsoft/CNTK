// This test tests all the methods in the C# collection wrapper

using System;
using li_std_vectorNamespace;

public class li_std_vector_runme {

  private static readonly int collectionSize = 20;
  private static readonly int midCollection = collectionSize/2;

  public static DoubleVector myDoubleVector;
  public static RealVector myRealVector;

  public static void Main() {
    // Setup collection
    DoubleVector vect = new DoubleVector();
    for (int i=0; i<collectionSize; i++) {
      double num = i*10.1;
      vect.Add(num);
    }

    // Count property test
    if (vect.Count != collectionSize)
      throw new Exception("Count test failed");

    // IsFixedSize property test
    if (vect.IsFixedSize)
      throw new Exception("IsFixedSize test failed");

    // IsReadOnly property test
    if (vect.IsReadOnly)
      throw new Exception("IsReadOnly test failed");

    // Item indexing
    vect[0] = 200.1;
    if (vect[0] != 200.1)
      throw new Exception("Item property test failed");
    vect[0] = 0*10.1;
    try {
      vect[-1] = 777.1;
      throw new Exception("Item out of range (1) test failed");
    } catch (ArgumentOutOfRangeException) {
    }
    try {
      vect[vect.Count] = 777.1;
      throw new Exception("Item out of range (2) test failed");
    } catch (ArgumentOutOfRangeException) {
    }

    // CopyTo() test
    {
      double[] outputarray = new double[collectionSize];
      vect.CopyTo(outputarray);
      int index = 0;
      foreach(double val in outputarray) {
        if (vect[index] != val)
          throw new Exception("CopyTo (1) test failed, index:" + index);
        index++;
      }
    }
    {
      double[] outputarray = new double[midCollection+collectionSize];
      vect.CopyTo(outputarray, midCollection);
      int index = midCollection;
      foreach(double val in vect) {
        if (outputarray[index] != val)
          throw new Exception("CopyTo (2) test failed, index:" + index);
        index++;
      }
    }
    {
      double[] outputarray = new double[3];
      vect.CopyTo(10, outputarray, 1, 2);
        if (outputarray[0] != 0.0 || outputarray[1] != vect[10] || outputarray[2] != vect[11])
          throw new Exception("CopyTo (3) test failed");
    }
    {
      double[] outputarray = new double[collectionSize-1];
      try {
        vect.CopyTo(outputarray);
        throw new Exception("CopyTo (4) test failed");
      } catch (ArgumentException) {
      }
    }
    {
      StructVector inputvector = new StructVector();
      int arrayLen = 10;
      for (int i=0; i<arrayLen; i++) {
        inputvector.Add(new Struct(i/10.0));
      }
      Struct[] outputarray = new Struct[arrayLen];
      inputvector.CopyTo(outputarray);
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num != inputvector[i].num)
          throw new Exception("CopyTo (6) test failed, i:" + i);
      }
      foreach (Struct s in inputvector) {
        s.num += 20.0;
      }
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num + 20.0 != inputvector[i].num )
          throw new Exception("CopyTo (7) test failed (only a shallow copy was made), i:" + i);
      }
    }
    {
      try {
        vect.CopyTo(null);
        throw new Exception("CopyTo (8) test failed");
      } catch (ArgumentNullException) {
      }
    }

    // Contains() test
    if (!vect.Contains(0*10.1))
      throw new Exception("Contains test 1 failed");
    if (!vect.Contains(10*10.1))
      throw new Exception("Contains test 2 failed");
    if (!vect.Contains(19*10.1))
      throw new Exception("Contains test 3 failed");
    if (vect.Contains(20*10.1))
      throw new Exception("Contains test 4 failed");

    {
      // ICollection constructor
      double[] doubleArray = new double[] { 0.0, 11.1, 22.2, 33.3, 44.4, 55.5, 33.3 };
      DoubleVector dv = new DoubleVector(doubleArray);
      if (doubleArray.Length != dv.Count)
        throw new Exception("ICollection constructor length check failed: " + doubleArray.Length + "-" + dv.Count);
      for (int i=0; i<doubleArray.Length; i++) {
        if (doubleArray[i] != dv[i])
          throw new Exception("ICollection constructor failed, index:" + i);
      }
      {
        Struct[] structArray = new Struct[] { new Struct(0.0), new Struct(11.1), new Struct(22.2), new Struct(33.3) };
        StructVector sv = new StructVector(structArray);
        for (int i=0; i<structArray.Length; i++) {
          structArray[i].num += 200.0;
        }
        for (int i=0; i<structArray.Length; i++) {
          if (structArray[i].num != sv[i].num + 200.0)
            throw new Exception("ICollection constructor not a deep copy, index:" + i);
        }
      }
      try {
        new DoubleVector((System.Collections.ICollection)null);
        throw new Exception("ICollection constructor null test failed");
      } catch (ArgumentNullException) {
      }
      {
        // Collection initializer test, requires C# 3.0
//        myDoubleVector = new DoubleVector() { 123.4, 567.8, 901.2 };
      }

      // IndexOf() test
      for (int i=0; i<collectionSize; i++) {
        if (vect.IndexOf(i*10.1) != i)
          throw new Exception("IndexOf test " + i + " failed");
      }
      if (vect.IndexOf(200.1) != -1)
        throw new Exception("IndexOf non-existent test failed");
      if (dv.IndexOf(33.3) != 3)
        throw new Exception("IndexOf position test failed");

      // LastIndexOf() test
      for (int i=0; i<collectionSize; i++) {
        if (vect.LastIndexOf(i*10.1) != i)
          throw new Exception("LastIndexOf test " + i + " failed");
      }
      if (vect.LastIndexOf(200.1) != -1)
        throw new Exception("LastIndexOf non-existent test failed");
      if (dv.LastIndexOf(33.3) != 6)
        throw new Exception("LastIndexOf position test failed");

      // Copy constructor test
      DoubleVector dvCopy = new DoubleVector(dv);
      for (int i=0; i<doubleArray.Length; i++) {
        if (doubleArray[i] != dvCopy[i])
          throw new Exception("Copy constructor failed, index:" + i);
      }
    }
    {
      // Repeat() test
      try {
        myDoubleVector = DoubleVector.Repeat(77.7, -1);
        throw new Exception("Repeat negative count test failed");
      } catch (ArgumentOutOfRangeException) {
      }
      DoubleVector dv = DoubleVector.Repeat(77.7, 5);
      if (dv.Count != 5)
        throw new Exception("Repeat count test failed");
      
      // Also tests enumerator
      {
        System.Collections.IEnumerator myEnumerator = dv.GetEnumerator();
        while ( myEnumerator.MoveNext() ) {
           if ((double)myEnumerator.Current != 77.7)
             throw new Exception("Repeat (1) test failed");
        }
      }
      {
        System.Collections.Generic.IEnumerator<double> myEnumerator = dv.GetEnumerator();
        while ( myEnumerator.MoveNext() ) {
           if (myEnumerator.Current != 77.7)
             throw new Exception("Repeat (2) test failed");
        }
      }
    }

    {
      // InsertRange() test
      DoubleVector dvect = new DoubleVector();
      for (int i=0; i<5; i++) {
        dvect.Add(1000.0*i);
      }
      vect.InsertRange(midCollection, dvect);
      if (vect.Count != collectionSize+dvect.Count)
        throw new Exception("InsertRange test size failed");

      for (int i=0; i<midCollection; i++) {
        if (vect.IndexOf(i*10.1) != i)
          throw new Exception("InsertRange (1) test " + i + " failed");
      }
      for (int i=0; i<dvect.Count; i++) {
        if (vect[i+midCollection] != dvect[i])
          throw new Exception("InsertRange (2) test " + i + " failed");
      }
      for (int i=midCollection; i<collectionSize; i++) {
        if (vect.IndexOf(i*10.1) != i+dvect.Count)
          throw new Exception("InsertRange (3) test " + i + " failed");
      }
      try {
        vect.InsertRange(0, null);
        throw new Exception("InsertRange (4) test failed");
      } catch (ArgumentNullException) {
      }

      // RemoveRange() test
      vect.RemoveRange(0, 0);
      vect.RemoveRange(midCollection, dvect.Count);
      if (vect.Count != collectionSize)
        throw new Exception("RemoveRange test size failed");
      for (int i=0; i<collectionSize; i++) {
        if (vect.IndexOf(i*10.1) != i)
          throw new Exception("RemoveRange test " + i + " failed");
      }
      try {
        vect.RemoveRange(-1, 0);
        throw new Exception("RemoveRange index out of range (1) test failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        vect.RemoveRange(0, -1);
        throw new Exception("RemoveRange count out of range (2) test failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        vect.RemoveRange(collectionSize+1, 0);
        throw new Exception("RemoveRange index and count out of range (1) test failed");
      } catch (ArgumentException) {
      }
      try {
        vect.RemoveRange(0, collectionSize+1);
        throw new Exception("RemoveRange index and count out of range (2) test failed");
      } catch (ArgumentException) {
      }

      // AddRange() test
      vect.AddRange(dvect);
      if (vect.Count != collectionSize+dvect.Count)
        throw new Exception("AddRange test size failed");
      for (int i=0; i<collectionSize; i++) {
        if (vect.IndexOf(i*10.1) != i)
          throw new Exception("AddRange (1) test " + i + " failed");
      }
      for (int i=0; i<dvect.Count; i++) {
        if (vect[i+collectionSize] != dvect[i])
          throw new Exception("AddRange (2) test " + i + " failed");
      }
      try {
        vect.AddRange(null);
        throw new Exception("AddRange (3) test failed");
      } catch (ArgumentNullException) {
      }
      vect.RemoveRange(collectionSize, dvect.Count);

      // GetRange() test
      int rangeSize = 5;
      DoubleVector returnedVec = vect.GetRange(0, 0);
      returnedVec = vect.GetRange(midCollection, rangeSize);
      if (returnedVec.Count != rangeSize)
        throw new Exception("GetRange test size failed");
      for (int i=0; i<rangeSize; i++) {
        if (returnedVec.IndexOf((i+midCollection)*10.1) != i)
          throw new Exception("GetRange test " + i + " failed");
      }
      try {
        vect.GetRange(-1, 0);
        throw new Exception("GetRange index out of range (1) test failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        vect.GetRange(0, -1);
        throw new Exception("GetRange count out of range (2) test failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        vect.GetRange(collectionSize+1, 0);
        throw new Exception("GetRange index and count out of range (1) test failed");
      } catch (ArgumentException) {
      }
      try {
        vect.GetRange(0, collectionSize+1);
        throw new Exception("GetRange index and count out of range (2) test failed");
      } catch (ArgumentException) {
      }
      {
        StructVector inputvector = new StructVector();
        int arrayLen = 10;
        for (int i=0; i<arrayLen; i++) {
          inputvector.Add(new Struct(i/10.0));
        }
        StructVector outputvector = inputvector.GetRange(0,arrayLen);
        for(int i=0; i<arrayLen; i++) {
          if (outputvector[i].num != inputvector[i].num)
            throw new Exception("GetRange (1) test failed, i:" + i);
        }
        foreach (Struct s in inputvector) {
          s.num += 20.0;
        }
        for(int i=0; i<arrayLen; i++) {
          if (outputvector[i].num + 20.0 != inputvector[i].num )
            throw new Exception("GetRange (2) test failed (only a shallow copy was made), i:" + i);
        }
      }
    }

    // Insert() test
    int pos = 0;
    int count = vect.Count;
    vect.Insert(pos, -5.1);
    count++;
    if (vect.Count != count || vect[pos] != -5.1)
      throw new Exception("Insert at beginning test failed");

    pos = midCollection;
    vect.Insert(pos, 85.1);
    count++;
    if (vect.Count != count || vect[pos] != 85.1)
      throw new Exception("Insert at " + pos + " test failed");

    pos = vect.Count;
    vect.Insert(pos, 195.1);
    count++;
    if (vect.Count != count || vect[pos] != 195.1)
      throw new Exception("Insert at end test failed");

    pos = vect.Count+1;
    try {
      vect.Insert(pos, 222.1); // should throw
      throw new Exception("Insert after end (1) test failed");
    } catch (ArgumentOutOfRangeException) {
    }
    if (vect.Count != count)
      throw new Exception("Insert after end (2) test failed");

    pos = -1;
    try {
      vect.Insert(pos, 333.1); // should throw
      throw new Exception("Insert before start (1) test failed");
    } catch (ArgumentOutOfRangeException) {
    }
    if (vect.Count != count)
      throw new Exception("Insert before start (2) test failed");

    // Remove() test
    vect.Remove(195.1);
    count--;
    vect.Remove(-5.1);
    count--;
    vect.Remove(85.1);
    count--;
    vect.Remove(9999.1); // element does not exist, should quietly do nothing
    if (vect.Count != count)
      throw new Exception("Remove count check test failed");
    for (int i=0; i<collectionSize; i++) {
      if (vect[i] != i*10.1)
        throw new Exception("Remove test failed, index:" + i);
    }

    // RemoveAt() test
    vect.Insert(0, -4.1);
    vect.Insert(midCollection, 84.1);
    vect.Insert(vect.Count, 194.1);
    vect.RemoveAt(vect.Count-1);
    vect.RemoveAt(midCollection);
    vect.RemoveAt(0);
    try {
      vect.RemoveAt(-1);
      throw new Exception("RemoveAt test (1) failed");
    } catch (ArgumentOutOfRangeException) {
    }
    try {
      vect.RemoveAt(vect.Count);
      throw new Exception("RemoveAt test (2) failed");
    } catch (ArgumentOutOfRangeException) {
    }
    for (int i=0; i<collectionSize; i++) {
      if (vect[i] != i*10.1)
        throw new Exception("RemoveAt test (3) failed, index:" + i);
    }

    {
      // Capacity test
      try {
        myDoubleVector = new DoubleVector(-1);
        throw new Exception("constructor setting capacity (1) test failed");
      } catch (ArgumentOutOfRangeException) {
      }

      DoubleVector dv = new DoubleVector(10);
      if (dv.Capacity != 10 || dv.Count != 0)
        throw new Exception("constructor setting capacity (2) test failed");
      dv.Capacity = 20;
      if (dv.Capacity != 20)
        throw new Exception("capacity test (1) failed");
      dv.Add(1.11);
      try {
        dv.Capacity = dv.Count-1;
        throw new Exception("capacity test (2) failed");
      } catch (ArgumentOutOfRangeException) {
      }

      // SetRange() test
      for (int i=dv.Count; i<collectionSize; i++) {
        dv.Add(0.0);
      }
      dv.SetRange(0, vect);
      if (dv.Count != collectionSize)
        throw new Exception("SetRange count check test failed");
      for (int i=0; i<collectionSize; i++) {
        if (vect[i] != dv[i])
          throw new Exception("SetRange test (1) failed, index:" + i);
      }
      try {
        dv.SetRange(-1, vect);
        throw new Exception("SetRange test (2) failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        dv.SetRange(1, vect);
        throw new Exception("SetRange test (3) failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        vect.SetRange(0, null);
        throw new Exception("SetRange (4) test failed");
      } catch (ArgumentNullException) {
      }

      // Reverse() test
      dv.Reverse();
      for (int i=0; i<collectionSize; i++) {
        if (vect[i] != dv[collectionSize-i-1])
          throw new Exception("Reverse test (1) failed, index:" + i);
      }
      dv.Reverse(0, collectionSize);
      for (int i=0; i<collectionSize; i++) {
        if (vect[i] != dv[i])
          throw new Exception("Reverse test (2) failed, index:" + i);
      }
      dv.Reverse(0, 0); // should do nothing!
      for (int i=0; i<collectionSize; i++) {
        if (vect[i] != dv[i])
          throw new Exception("Reverse test (3) failed, index:" + i);
      }
      try {
        dv.Reverse(-1, 0);
        throw new Exception("Reverse test (4) failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        dv.Reverse(0, -1);
        throw new Exception("Reverse test (5) failed");
      } catch (ArgumentOutOfRangeException) {
      }
      try {
        dv.Reverse(collectionSize+1, 0);
        throw new Exception("Reverse test (6) failed");
      } catch (ArgumentException) {
      }
      try {
        dv.Reverse(0, collectionSize+1);
        throw new Exception("Reverse test (7) failed");
      } catch (ArgumentException) {
      }
    }

    // foreach test
    {
      int index=0;
      foreach (double s in vect) {
        if (s != index*10.1)
          throw new Exception("foreach test failed, index:" + index);
        index++;
      }
    }

    // Clear() test
    vect.Clear();
    if (vect.Count != 0)
      throw new Exception("Clear failed");

    // Finally test the methods being wrapped
    {
      IntVector iv = new IntVector();
      for (int i=0; i<4; i++) {
        iv.Add(i);
      }

      double x = li_std_vector.average(iv);
      x += li_std_vector.average( new IntVector( new int[] {1, 2, 3, 4} ) );
      myRealVector = li_std_vector.half( new RealVector( new float[] {10F, 10.5F, 11F, 11.5F} ) );

      DoubleVector dvec = new DoubleVector();
      for (int i=0; i<10; i++) {
        dvec.Add(i/2.0);
      }
      li_std_vector.halve_in_place(dvec);
    }

    // Dispose()
    {
      using (StructVector vs = new StructVector( new Struct[] { new Struct(0.0), new Struct(11.1) } ) )
      using (DoubleVector vd = new DoubleVector( new double[] { 0.0, 11.1 } ) ) {
      }
    }

    // More wrapped methods
    {
      RealVector v0 = li_std_vector.vecreal(new RealVector());
      float flo = 123.456f;
      v0.Add(flo);
      flo = v0[0];

      IntVector v1 = li_std_vector.vecintptr(new IntVector());
      IntPtrVector v2 = li_std_vector.vecintptr(new IntPtrVector());
      IntConstPtrVector v3 = li_std_vector.vecintconstptr(new IntConstPtrVector());

      v1.Add(123);
      v2.Clear();
      v3.Clear();

      StructVector v4 = li_std_vector.vecstruct(new StructVector());
      StructPtrVector v5 = li_std_vector.vecstructptr(new StructPtrVector());
      StructConstPtrVector v6 = li_std_vector.vecstructconstptr(new StructConstPtrVector());

      v4.Add(new Struct(123));
      v5.Add(new Struct(123));
      v6.Add(new Struct(123));
    }

    // Test vectors of pointers
    {
      StructPtrVector inputvector = new StructPtrVector();
      int arrayLen = 10;
      for (int i=0; i<arrayLen; i++) {
        inputvector.Add(new Struct(i/10.0));
      }
      Struct[] outputarray = new Struct[arrayLen];
      inputvector.CopyTo(outputarray);
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num != inputvector[i].num)
          throw new Exception("StructPtrVector test (1) failed, i:" + i);
      }
      foreach (Struct s in inputvector) {
        s.num += 20.0;
      }
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num != 20.0 + i/10.0)
          throw new Exception("StructPtrVector test (2) failed (a deep copy was incorrectly made), i:" + i);
      }

      int rangeSize = 5;
      int mid = arrayLen/2;
      StructPtrVector returnedVec = inputvector.GetRange(mid, rangeSize);
      for (int i=0; i<rangeSize; i++) {
        if (inputvector[i+mid].num != returnedVec[i].num)
          throw new Exception("StructPtrVector test (3) failed, i:" + i);
      }
    }

    // Test vectors of const pointers
    {
      StructConstPtrVector inputvector = new StructConstPtrVector();
      int arrayLen = 10;
      for (int i=0; i<arrayLen; i++) {
        inputvector.Add(new Struct(i/10.0));
      }
      Struct[] outputarray = new Struct[arrayLen];
      inputvector.CopyTo(outputarray);
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num != inputvector[i].num)
          throw new Exception("StructConstPtrVector test (1) failed, i:" + i);
      }
      foreach (Struct s in inputvector) {
        s.num += 20.0;
      }
      for(int i=0; i<arrayLen; i++) {
        if (outputarray[i].num != 20.0 + i/10.0)
          throw new Exception("StructConstPtrVector test (2) failed (a deep copy was incorrectly made), i:" + i);
      }

      int rangeSize = 5;
      int mid = arrayLen/2;
      StructConstPtrVector returnedVec = inputvector.GetRange(mid, rangeSize);
      for (int i=0; i<rangeSize; i++) {
        if (inputvector[i+mid].num != returnedVec[i].num)
          throw new Exception("StructConstPtrVector test (3) failed, i:" + i);
      }
    }

  }

}

