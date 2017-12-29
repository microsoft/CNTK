// This test tests all the methods in the C# collection wrapper
using System;
using cpp11_li_std_arrayNamespace;

public class cpp11_li_std_array_runme
{
    private static ArrayInt6 ToArray6(int[] a)
    {
        if (a.Length != 6)
            throw new Exception("a is incorrect size");
        return new ArrayInt6(a);
    }

    private static void compareContainers(ArrayInt6 actual, int[] expected)
    {
        if (actual.Count != expected.Length)
            throw new Exception("Sizes are different: " + actual.Count + " " + expected.Length);
        for (int i=0; i<actual.Count; ++i)
        {
            int actualValue = actual[i];
            int expectedValue = expected[i];
            if (actualValue != expectedValue)
                throw new Exception("Value is wrong for element " + i + ". Expected " + expectedValue + " got: " + actualValue);
        }
        if (actual.IsEmpty)
            throw new Exception("ai should not be empty");
    }

    static void Main()
    {
        ArrayInt6 ai = new ArrayInt6();
        compareContainers(ai, new int[] { 0, 0, 0, 0, 0, 0 });

        int[] vals = { 10, 20, 30, 40, 50, 60 };
        for (int i = 0; i < ai.Count; ++i)
            ai[i] = vals[i];
        compareContainers(ai, vals);

        // Check return
        compareContainers(cpp11_li_std_array.arrayOutVal(), new int[] { -2, -1, 0, 0, 1, 2 });
        compareContainers(cpp11_li_std_array.arrayOutConstRef(), new int[] { -2, -1, 0, 0, 1, 2 });
        compareContainers(cpp11_li_std_array.arrayOutRef(), new int[] { -2, -1, 0, 0, 1, 2 });
        compareContainers(cpp11_li_std_array.arrayOutPtr(), new int[] { -2, -1, 0, 0, 1, 2 });

        // Check passing arguments
        ai = cpp11_li_std_array.arrayInVal(ToArray6(new int[] { 9, 8, 7, 6, 5, 4 }));
        compareContainers(ai, new int[] { 90, 80, 70, 60, 50, 40 });

        ai = cpp11_li_std_array.arrayInConstRef(ToArray6(new int[] { 9, 8, 7, 6, 5, 4 }));
        compareContainers(ai, new int[] { 90, 80, 70, 60, 50, 40 });

        ai = new ArrayInt6(ToArray6(new int[] { 9, 8, 7, 6, 5, 4 }));
        cpp11_li_std_array.arrayInRef(ai);
        compareContainers(ai, new int[] { 90, 80, 70, 60, 50, 40 });

        ai = new ArrayInt6(ToArray6(new int[] { 9, 8, 7, 6, 5, 4 }));
        cpp11_li_std_array.arrayInPtr(ai);
        compareContainers(ai, new int[] { 90, 80, 70, 60, 50, 40 });

        // fill
        ai.Fill(111);
        compareContainers(ai, new int[] { 111, 111, 111, 111, 111, 111 });

        // out of range errors
        try
        {
            ai[ai.Count] = 0;
            throw new Exception("Out of range exception not caught");
        }
        catch (ArgumentOutOfRangeException)
        {
        }
        try
        {
            ai[-1] = 0;
            throw new Exception("Out of range exception not caught");
        }
        catch (ArgumentOutOfRangeException)
        {
        }
    }
}
