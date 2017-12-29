
/***********************************************************************************************
   NOTE: This is a custom testcase and should be run using make intermediary_classname.customtest
 ***********************************************************************************************/

using System;
using intermediary_classnameNamespace;

public class runme
{
    static void Main() 
    {
      // test the renamed module class is correctly named
      double d = intermediary_classnameModule.maxdouble(10.0, 20.0);
      if (d!=20.0) throw new Exception("Test failed");

      // test the renamed intermediary class is correctly named
      IntPtr ptr = intermediary_classname.new_vecdouble(10);
      intermediary_classname.delete_vecdouble(new System.Runtime.InteropServices.HandleRef(null,ptr));
    }
}
