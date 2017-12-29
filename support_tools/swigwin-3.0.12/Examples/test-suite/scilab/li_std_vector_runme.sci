exec("swigtest.start", -1);

// TODO: support for STL vectors operator =
iv = new_DoubleVector();
//for i=1:4
//    iv(i) = i;
//end
//x = average(iv);

//if x <> 2.5 then swigtesterror(); end
exit
exec("swigtest.quit", -1);
