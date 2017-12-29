import overload_rename


f = overload_rename.Foo(1)
f = overload_rename.Foo(1, 1)
f = overload_rename.Foo_int(1, 1)
f = overload_rename.Foo_int(1, 1, 1)
