require("import")	-- the import fn
-- need to load two modules
import("imports_a")	-- import code
import("imports_b")	-- import code

b=imports_b.B()
b:hello() -- call member function in A which is in a different SWIG generated library.
b:bye()

assert (b:member_virtual_test(imports_a.A_memberenum1) == imports_a.A_memberenum2)
assert (b:global_virtual_test(imports_a.globalenum1) == imports_a.globalenum2)

imports_b.global_test(imports_a.A_memberenum1)

--[[    B b = new B();
    b.hello(); //call member function in A which is in a different SWIG generated library.

            B b = new B();
        b.hello(); //call member function in A which is in a different SWIG generated library.
        b.bye();

        if (b.member_virtual_test(A.MemberEnum.memberenum1) != A.MemberEnum.memberenum2)
          throw new Exception("Test 1 failed");
        if (b.global_virtual_test(GlobalEnum.globalenum1) != GlobalEnum.globalenum2)
          throw new Exception("Test 2 failed");

        imports_b.global_test(A.MemberEnum.memberenum1);
]]
