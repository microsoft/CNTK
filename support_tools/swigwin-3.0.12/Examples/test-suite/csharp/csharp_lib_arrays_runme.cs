using System;
using csharp_lib_arraysNamespace;

public class runme
{
  static void Main() 
  {
    {
      int[] source = { 1, 2, 3, 4, 5 };
      int[] target = new int[ source.Length ];

      csharp_lib_arrays.myArrayCopy( source, target, target.Length );
      CompareArrays(source, target);
    }

    {
      int[] source = { 1, 2, 3, 4, 5 };
      int[] target = new int[ source.Length ];

      csharp_lib_arrays.myArrayCopyUsingFixedArrays( source, target, target.Length );
      CompareArrays(source, target);
    }

    {
      int[] source = { 1, 2, 3, 4, 5 };
      int[] target = new int[] { 6, 7, 8, 9, 10 };

      csharp_lib_arrays.myArraySwap( source, target, target.Length );

      for (int i=0; i<target.Length; ++i)
        target[i] += 5;
      CompareArrays(source, target);
    }

    {
      int[] source = { 1, 2, 3, 4, 5 };
      int[] target = new int[] { 6, 7, 8, 9, 10 };

      csharp_lib_arrays.myArraySwapUsingFixedArrays( source, target, target.Length );

      for (int i=0; i<target.Length; ++i)
        target[i] += 5;
      CompareArrays(source, target);
    }
  }
  
  static void CompareArrays( int[] a, int[] b ) 
  {
    if (a.Length != b.Length)
      throw new Exception("size mismatch");

    for(int i=0; i<a.Length; ++i) {
      if (a[i] != b[i]) {
        Console.Error.WriteLine("a:");
        PrintArray(a);
        Console.Error.WriteLine("b:");
        PrintArray(b);
        throw new Exception("element mismatch");
      }
    }
  }

  static void PrintArray( int[] a ) 
  {
    foreach ( int i in a ) 
      Console.Error.Write( "{0} ", i );
    Console.Error.WriteLine();
  }
}

