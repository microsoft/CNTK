exec("swigtest.start", -1);

// OUTPUT

[a, b] = output2();
checkequal(a, 1, "[a, b] = output2(): a");
checkequal(b, 2, "[a, b] = output2(): b");

[ret, a, b] = output2Ret();
checkequal(ret, 3, "[a, b] = output2Ret(): b");
checkequal(a, 1, "[a, b] = output2Ret(): a");
checkequal(b, 2, "[a, b] = output2Ret(): b");

[c, d] = output2Input2(1, 2);
checkequal(c, 2, "[c, d] = output2Input2(1, 2): c");
checkequal(d, 4, "[c, d] = output2Input2(1, 2): d");

[ret, c, d] = output2Input2Ret(1, 2);
checkequal(ret, 6, "[ret, c, d] = output2Input2Ret(1, 2): ret");
checkequal(c, 2, "[ret, c, d] = output2Input2Ret(1, 2): c");
checkequal(d, 4, "[ret, c, d = output2Input2Ret(1, 2): d");

[ret, a, b, c] = output3Input1Ret(10);
checkequal(ret, 10, "[ret, a, b, c] = output3Input1Ret(10): ret");
checkequal(a, 11, "[ret, a, b, c] = output3Input1Ret(10): a");
checkequal(b, 12, "[ret, a, b, c] = output3Input1Ret(10): b");
checkequal(c, 13, "[ret, a, b, c] = output3Input1Ret(10): c");

[ret, a, b, c] = output3Input3Ret(10, 20, 30);
checkequal(ret, 66, "[ret, a, b, c] = output3Input1Ret(10, 20, 30): ret");
checkequal(a, 11, "[ret, a, b, c] = output3Input1Ret(10, 20, 30): a");
checkequal(b, 22, "[ret, a, b, c] = output3Input1Ret(10, 20, 30): b");
checkequal(c, 33, "[ret, a, b, c] = output3Input1Ret(10, 20, 30): c");


// INOUT

[a, b] = inout2(1, 2);
checkequal(a, 2, "[a, b] = output2(1, 2): a");
checkequal(b, 4, "[a, b] = output2(1, 2): b");

[ret, a, b] = inout2Ret(1, 2);
checkequal(ret, 6, "[a, b] = inout2Ret(1, 2): b");
checkequal(a, 2, "[a, b] = inout2Ret(1, 2): a");
checkequal(b, 4, "[a, b] = inout2Ret(1, 2): b");

[c, d] = inout2Input2(1, 2, 1, 1);
checkequal(c, 2, "[c, d] = inout2Input2(1, 2): c");
checkequal(d, 3, "[c, d] = inout2Input2(1, 2): d");

[ret, c, d] = inout2Input2Ret(1, 2, 1, 1);
checkequal(ret, 5, "[c, d] = inout2Input2Ret(1, 2): ret");
checkequal(c, 2, "[c, d] = inout2Input2Ret(1, 2): c");
checkequal(d, 3, "[c, d] = inout2Input2Ret(1, 4): d");

[ret, a, b, c] = inout3Input1Ret(10, 1, 2, 3);
checkequal(ret, 10, "[ret, a, b, c] = output3Input1Ret(ret, 1, 2, 3): ret");
checkequal(a, 11, "[ret, a, b, c] = output3Input1Ret(ret, 1, 2, 3): a");
checkequal(b, 12, "[ret, a, b, c] = output3Input1Ret(ret, 1, 2, 3): b");
checkequal(c, 13, "[ret, a, b, c] = output3Input1Ret(ret, 1, 2, 3): c");

[ret, a, b, c] = inout3Input3Ret(10, 1, 20, 2, 30, 3);
checkequal(ret, 66, "[ret, a, b, c] = output3Input1Ret(10, 20, 30): ret");
checkequal(a, 11, "[ret, a, b, c] = inout3Input1Ret(10, 1, 20, 2, 30, 3): a");
checkequal(b, 22, "[ret, a, b, c] = inout3Input1Ret(10, 1, 20, 2, 30, 3): b");
checkequal(c, 33, "[ret, a, b, c] = inout3Input1Ret(10, 1, 20, 2, 30, 3): c");


// CLASS

a = new_ClassA();

[ret, c, d] = ClassA_output2Input2Ret(a, 1, 2);
checkequal(ret, 6, "[ret, c, d] = ClassA_output2Input2Ret(a, 1, 2): ret");
checkequal(c, 2, "[c, d] = ClassA_output2Input2Ret(a, 1, 2): c");
checkequal(d, 4, "[c, d] = ClassA_output2Input2Ret(a, 1, 2): d");

[ret, c, d] = ClassA_inout2Input2Ret(a, 1, 2, 1, 1);
checkequal(ret, 5, "[ret, c, d] = ClassA_inout2Input2Ret(a, 1, 2): ret");
checkequal(c, 2, "[c, d] = ClassA_inout2Input2(a, 1, 2): c");
checkequal(d, 3, "[c, d] = ClassA_inout2Input2(a, 1, 2): d");

delete_ClassA(a);


exec("swigtest.quit", -1);

