lines(0);
ilib_verbose(0);
ierr = exec('loader.sce', 'errcatch');
if ierr <> 0 then
  disp(lasterror());
  exit(ierr);
end

// Test lib double matrix functions
disp("Call lib function getDoubleMatrix()");
doubleMatrix = getDoubleMatrix();
disp(doubleMatrix);

disp("Call lib function sumDoubleMatrix()");
s = sumDoubleMatrix(doubleMatrix);
disp(s);

disp("Call lib function squareDoubleMatrix()");
sqrd = squareDoubleMatrix(doubleMatrix);
disp(sqrd);


// Test lib integer matrix functions

disp("Call lib function getIntegerMatrix()");
integerMatrix = getIntegerMatrix();
disp(integerMatrix);

disp("Call lib function sumIntegerMatrix()");
s = sumIntegerMatrix(integerMatrix);
disp(s);

disp("Call lib function squareIntegerMatrix()");
sqri = squareIntegerMatrix(integerMatrix);
disp(sqri);


// Test lib string matrix functions

disp("Call lib function getStringVector()");
stringVector = getStringVector();
disp(stringVector);

disp("Call lib function concatStringVector()");
stringVector2 = concatStringVector(stringVector);
disp(stringVector2);

exit



