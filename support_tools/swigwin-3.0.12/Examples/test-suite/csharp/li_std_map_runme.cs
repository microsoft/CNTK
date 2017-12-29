/* -----------------------------------------------------------------------------
 * li_std_map_runme.cs
 *
 * SWIG C# tester for std_map.i
 * This class tests all the functionality of the std_map.i wrapper.
 * Upon successful testing, the main function doesn't print out anything.
 * If any error is found - it will be printed on the screen.
 * ----------------------------------------------------------------------------- */

using System;
using System.Collections.Generic;
using li_std_mapNamespace;

public class li_std_map_runme {

    private static readonly int collectionSize = 20;
    private static readonly int midCollection = collectionSize / 2;

    public static void Main()
    {
        // Set up an int int map
        StringIntMap simap = new StringIntMap();
        for (int i = 0; i < collectionSize; i++)
        {
            int val = i * 18;
            simap.Add(i.ToString(), val);
        }

        // Count property test
        if (simap.Count != collectionSize)
            throw new Exception("Count test failed");

        // IsReadOnly property test
        if (simap.IsReadOnly)
            throw new Exception("IsReadOnly test failed");

        // Item indexing test
        simap["0"] = 200;
        if (simap["0"] != 200)
            throw new Exception("Item property test failed");
        simap["0"] = 0 * 18;

        // ContainsKey() test
        for (int i = 0; i < collectionSize; i++)
        {
            if (!simap.ContainsKey(i.ToString()))
                throw new Exception("ContainsKey test " + i + " failed");
        }

        // ContainsKey() test
        for (int i = 0; i < collectionSize; i++)
        {
            if (!simap.Contains(new KeyValuePair<string, int>(i.ToString(), i * 18)))
                throw new Exception("Contains test " + i + " failed");
        }

        // TryGetValue() test
        int value;
        bool rc = simap.TryGetValue("3", out value);
        if (rc != true || value != (3 * 18))
            throw new Exception("TryGetValue test 1 failed");

        rc = simap.TryGetValue("-1", out value);
        if (rc != false)
            throw new Exception("TryGetValue test 2 failed");

        // Keys and Values test
        {
            IList<string> keys = new List<string>(simap.Keys);
            IList<int> values = new List<int>(simap.Values);
            Dictionary<string, int> check = new Dictionary<string, int>();
            if (keys.Count != collectionSize)
                throw new Exception("Keys count test failed");

            if (values.Count != collectionSize)
                throw new Exception("Values count test failed");

            for (int i = 0; i < keys.Count; i++)
            {
                if (simap[keys[i]] != values[i])
                    throw new Exception("Keys and values test failed for index " + i);
                check.Add(keys[i], values[i]);
            }

            for (int i = 0; i < collectionSize; i++)
            {
                if (!check.ContainsKey(i.ToString()))
                  throw new Exception("Keys and Values ContainsKey test " + i + " failed");
            }
        }

        // Add and Remove test
        for (int i = 100; i < 103; i++)
        {
            simap.Add(i.ToString(), i * 18);
            if (!simap.ContainsKey(i.ToString()) || simap[i.ToString()] != (i * 18))
                throw new Exception("Add test failed for index " + i);

            simap.Remove(i.ToString());
            if (simap.ContainsKey(i.ToString()))
                throw new Exception("Remove test failed for index " + i);
        }

        for (int i = 200; i < 203; i++)
        {
            simap.Add(new KeyValuePair<string, int>(i.ToString(), i * 18));
            if (!simap.ContainsKey(i.ToString()) || simap[i.ToString()] != (i * 18))
                throw new Exception("Add explicit test failed for index " + i);

            simap.Remove(new KeyValuePair<string, int>(i.ToString(), i * 18));
            if (simap.ContainsKey(i.ToString()))
                throw new Exception("Remove explicit test failed for index " + i);
        }

        // Duplicate key test
        try
        {
            simap.Add("3", 0);
            throw new Exception("Adding duplicate key test failed");
        }
        catch (ArgumentException)
        {
        }

        // CopyTo() test
        {
            KeyValuePair<string, int>[] outputarray = new KeyValuePair<string, int>[collectionSize];
            simap.CopyTo(outputarray);
            foreach (KeyValuePair<string, int> val in outputarray)
            {
                if (simap[val.Key] != val.Value)
                    throw new Exception("CopyTo (1) test failed, index:" + val.Key);
            }
        }
        {
            KeyValuePair<string, int>[] outputarray = new KeyValuePair<string, int>[midCollection + collectionSize];
            simap.CopyTo(outputarray, midCollection);
            for (int i = midCollection; i < midCollection + collectionSize; i++)
            {
                KeyValuePair<string, int> val = outputarray[i];
                if (simap[val.Key] != val.Value)
                    throw new Exception("CopyTo (2) test failed, index:" + val.Key);
            }
        }
        {
            KeyValuePair<string, int>[] outputarray = new KeyValuePair<string, int>[collectionSize - 1];
            try
            {
                simap.CopyTo(outputarray);
                throw new Exception("CopyTo (4) test failed");
            }
            catch (ArgumentException)
            {
            }
        }

        // Clear test
        simap.Clear();
        if (simap.Count != 0)
            throw new Exception("Clear test failed");

        // Test wrapped methods
        for (int i = 1; i <= 5; i++)
        {
            simap[i.ToString()] = i;
        }
        double avg = li_std_map.valueAverage(simap);
        if (avg != 3.0)
            throw new Exception("Wrapped method valueAverage test failed. Got " + avg);

        string keyStringified = li_std_map.stringifyKeys(simap);
        if (keyStringified != " 1 2 3 4 5")
            throw new Exception("Wrapped method stringifyKeys test failed. Got " + keyStringified);

        // Test a map with a new complex type (Struct)
        {
            IntStructMap ismap = new IntStructMap();
            for (int i = 0; i < 10; i++)
            {
                ismap.Add(i, new Struct(i * 10.1));
            }

            if (ismap.Count != 10)
                throw new Exception("Count test on complex type map failed");

            foreach (KeyValuePair<int, Struct> p in ismap)
            {
                if ((p.Key * 10.1) != p.Value.num)
                    throw new Exception("Iteration test on complex type map failed for index " + p.Key);
            }
        }

        // Test a map of pointers
        {
            IntStructPtrMap ispmap = new IntStructPtrMap();
            for (int i = 0; i < 10; i++)
            {
                ispmap.Add(i, new Struct(i * 10.1));
            }

            if (ispmap.Count != 10)
                throw new Exception("Count test on complex type pointer map failed");

            foreach (KeyValuePair<int, Struct> p in ispmap)
            {
                if ((p.Key * 10.1) != p.Value.num)
                    throw new Exception("Iteration test on complex type pointer map failed for index " + p.Key);
            }
        }
        {
            IntStructConstPtrMap iscpmap = new IntStructConstPtrMap();
            for (int i = 0; i < 10; i++)
            {
                iscpmap.Add(i, new Struct(i * 10.1));
            }

            if (iscpmap.Count != 10)
                throw new Exception("Count test on complex type const pointer map failed");

            foreach (KeyValuePair<int, Struct> p in iscpmap)
            {
                if ((p.Key * 10.1) != p.Value.num)
                    throw new Exception("Iteration test on complex type const pointer map failed for index " + p.Key);
            }
        }

        // Test complex type as key (Struct)
        {
            StructIntMap limap = new StructIntMap();
            Struct s7 = new Struct(7);
            Struct s8 = new Struct(8);
            limap[s7] = 8;
            if (limap[s7] != 8)
                throw new Exception("Assignment test on complex key map failed");

            if (!limap.ContainsKey(s7))
                throw new Exception("Key test (1) on complex key map failed");

            if (limap.ContainsKey(s8))
                throw new Exception("Key test (2) on complex key map failed");
        }

        // All done
    }
}

