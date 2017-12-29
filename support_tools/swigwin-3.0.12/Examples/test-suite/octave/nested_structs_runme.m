nested_structs

named = nested_structs.Named();
named.val = 999;
assert(nested_structs.nestedByVal(named), 999);
assert(nested_structs.nestedByPtr(named), 999);

outer = nested_structs.Outer();
outer.inside1.val = 456;
assert(nested_structs.getInside1Val(outer), 456);

outer.inside1 = named;
assert(nested_structs.getInside1Val(outer), 999);

