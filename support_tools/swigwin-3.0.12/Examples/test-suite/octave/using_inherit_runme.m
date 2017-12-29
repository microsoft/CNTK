using_inherit

b = Bar();
if (b.test(3) != 3)
  error("Bar::test(int)")
endif

if (b.test(3.5) != 3.5)
  error("Bar::test(double)")
endif


b = Bar2();
if (b.test(3) != 6)
  error("Bar2::test(int)")
endif

if (b.test(3.5) != 7.0)
  error("Bar2::test(double)")
endif


b = Bar3();
if (b.test(3) != 6)
  error("Bar3::test(int)")
endif

if (b.test(3.5) != 7.0)
  error("Bar3::test(double)")
endif


b = Bar4();
if (b.test(3) != 6)
  error("Bar4::test(int)")
endif

if (b.test(3.5) != 7.0)
  error("Bar4::test(double)")
endif


b = Fred1();
if (b.test(3) != 3)
  error("Fred1::test(int)")
endif

if (b.test(3.5) != 7.0)
  error("Fred1::test(double)")
endif


b = Fred2();
if (b.test(3) != 3)
  error("Fred2::test(int)")
endif

if (b.test(3.5) != 7.0)
  error("Fred2::test(double)")
endif

