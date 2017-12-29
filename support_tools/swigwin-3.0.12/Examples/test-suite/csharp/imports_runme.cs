using System;
using importsNamespace;

public class runme
{
    static void Main() {
        B b = new B();
        b.hello(); //call member function in A which is in a different SWIG generated library.
        b.bye();

        if (b.member_virtual_test(A.MemberEnum.memberenum1) != A.MemberEnum.memberenum2)
          throw new Exception("Test 1 failed");
        if (b.global_virtual_test(GlobalEnum.globalenum1) != GlobalEnum.globalenum2)
          throw new Exception("Test 2 failed");

        imports_b.global_test(A.MemberEnum.memberenum1);
    }
}
