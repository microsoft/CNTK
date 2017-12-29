extend_placement

foo = extend_placement.Foo();
foo = extend_placement.Foo(1);
foo = extend_placement.Foo(1,1);
foo.spam();
foo.spam("hello");
foo.spam(1);
foo.spam(1,1);
foo.spam(1,1,1);
foo.spam(extend_placement.Foo());
foo.spam(extend_placement.Foo(), 1.0);


bar = extend_placement.Bar();
bar = extend_placement.Bar(1);
bar.spam();
bar.spam("hello");
bar.spam(1);
bar.spam(1,1);
bar.spam(1,1,1);
bar.spam(extend_placement.Bar());
bar.spam(extend_placement.Bar(), 1.0);


foo = extend_placement.FooTi();
foo = extend_placement.FooTi(1);
foo = extend_placement.FooTi(1,1);
foo.spam();
foo.spam("hello");
foo.spam(1);
foo.spam(1,1);
foo.spam(1,1,1);
foo.spam(extend_placement.Foo());
foo.spam(extend_placement.Foo(), 1.0);


bar = extend_placement.BarTi();
bar = extend_placement.BarTi(1);
bar.spam();
bar.spam("hello");
bar.spam(1);
bar.spam(1,1);
bar.spam(1,1,1);
bar.spam(extend_placement.Bar());
bar.spam(extend_placement.Bar(), 1.0);
