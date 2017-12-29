exec("swigtest.start", -1);

function checkPair(pair, expected_first, expected_second, func)
  checkequal(IntPair_first_get(pair), expected_first, func + ": first");;
  checkequal(IntPair_second_get(pair), expected_second, func + ": second");;
endfunction

intPair = makeIntPair(7, 6);
checkPair(intPair, 7, 6, "makeIntPair()");

intPairPtr = makeIntPairPtr(7, 6);
checkPair(intPairPtr, 7, 6, "makeIntPairPtr()");

intPairRef = makeIntPairRef(7, 6);
checkPair(intPairRef, 7, 6, "makeIntPairRef()");

intPairConstRef = makeIntPairConstRef(7, 6);
checkPair(intPairConstRef, 7, 6, "makeIntPairConstRef()");

// call fns
checkequal(product1(intPair), 42, "product1(intPair)");
checkequal(product2(intPair), 42, "product2(intPair)");
checkequal(product3(intPair), 42, "product3(intPair)")

// also use the pointer version
checkequal(product1(intPairPtr), 42, "product1(intPairPtr)");
checkequal(product2(intPairPtr), 42, "product2(intPairPtr)");
checkequal(product3(intPairPtr), 42, "product3(intPairPtr)");

// or the other types
checkequal(product1(intPairRef), 42, "product1(intPairRef)");
checkequal(product2(intPairRef), 42, "product2(intPairRef)");
checkequal(product3(intPairRef), 42, "product3(intPairRef)");
checkequal(product1(intPairConstRef), 42, "product3(intPairConstRef)");
checkequal(product2(intPairConstRef), 42, "product2(intPairConstRef)");
checkequal(product3(intPairConstRef), 42, "product1(intPairConstRef)");

exec("swigtest.quit", -1);
