exec("swigtest.start", -1);

derived = new_Derived();

checkequal(test(derived), 0, "test(derived)");
checkequal(test2(derived), 1, "test2(derived)");

delete_Derived(derived);

exec("swigtest.quit", -1);

