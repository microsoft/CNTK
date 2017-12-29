exec("swigtest.start", -1);

i = new_iFoo();
checkequal(iFoo_blah_test(i, 4), 4, "iFoo_blah_test(i, 4) test fails");
checkequal(iFoo_spam_test(i, 5), 5, "iFoo_spam_test(i, 5) test fails");
checkequal(iFoo_groki_test(i, 6), 6, "iFoo_groki_test(i, 6) test fails");
delete_iFoo(i);

d = new_dFoo();
checkequal(dFoo_blah_test(d, 4), 4, "dFoo_blah_test(d, 4) test fails");
checkequal(dFoo_spam(d, 5), 5, "dFoo_spam_test(d, 5) test fails");
checkequal(dFoo_grok_test(d, 6), 6, "dFoo_groki_test(d, 6) test fails");
delete_dFoo(d);

exec("swigtest.quit", -1);
