exec("swigtest.start", -1);

try
    a = get_test();
catch
    swigtesterror();
end

// Test default values
checkequal(test_myInt_get(a), 100, "test_myInt_get() test fails.");
checkequal(test_myShort_get(a), 200, "test_myShort_get() test fails.");

// Write new values
try
    test_myInt_set(a, 42)
    test_myShort_set(a, 12)
catch
    swigtesterror();
end

// Read new values
checkequal(test_myInt_get(a), 42, "test_myInt_get() test fails.");
checkequal(test_myShort_get(a), 12, "test_myShort_get() test fails.");

// Destroy pointer
delete_test(a);

exec("swigtest.quit", -1);
