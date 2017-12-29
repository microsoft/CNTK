// test matrix.i library

exec("swigtest.start", -1);

// test matrix passed as output argument from fonction
function test_outMatrix(func, valueType, expectedOutMatrix)
  funcName = msprintf("out%s%s", valueType, func);
  cmd = msprintf("outMatrix = %s();", funcName);
  ierr = execstr(cmd, "errcatch");
  if ierr <> 0 then
    swigtesterror(msprintf("Error %d in %s", ierr, funcName));
  end
  checkequal(outMatrix, expectedOutMatrix, funcName);
endfunction

// test matrix passed as input argument of fonction
function test_inMatrix(func, valueType, inMatrix, expectedInValue)
  funcName = msprintf("in%s%s", valueType, func);
  cmd = msprintf("inValue = %s(inMatrix);", funcName);
  ierr = execstr(cmd, "errcatch");
  if ierr <> 0 then
    swigtesterror(msprintf("Error %d in %s", ierr, funcName));
  end
  checkequal(inValue, expectedInValue, funcName);
endfunction

// test matrixes passed as input and output arguments of fonction
function test_inoutMatrix(func, valueType, inoutMatrix, expectedInoutMatrix)
  funcName = msprintf("inout%s%s", valueType, func);
  cmd = msprintf("inoutMatrix = %s(inoutMatrix);", funcName);
  ierr = execstr(cmd, "errcatch");
  if ierr <> 0 then
    swigtesterror(msprintf("Error %d in %s", ierr, funcName));
  end
  checkequal(inoutMatrix, expectedInoutMatrix, funcName);
endfunction

function test_matrix_typemaps(valueType, ..
  expectedOutMatrixDims, expectedOutMatrixSize, ..
  expectedInValue, ..
  expectedInoutMatrixDims, expectedInoutMatrixSize)

  test_outMatrix("MatrixDims", valueType, expectedOutMatrixDims);
  test_outMatrix("MatrixSize", valueType, expectedOutMatrixSize);
  matrixDims = expectedOutMatrixDims;
  matrixSize = expectedOutMatrixSize;
  test_inMatrix("MatrixDims", valueType, matrixDims, expectedInValue);
  test_inMatrix("MatrixSize", valueType, matrixSize, expectedInValue);
  test_inoutMatrix("MatrixDims", valueType, matrixDims, expectedInoutMatrixDims);
  test_inoutMatrix("MatrixSize", valueType, matrixSize, expectedInoutMatrixSize);
endfunction


m = [0  3;  1  4;  2  5];
v = [0  1   2  3   4  5];
test_matrix_typemaps("Int", m, v, sum(m), m .* m, v .* v);
test_matrix_typemaps("Double", m, v, sum(m), m .* m, v .* v);

m = ["A" "D"; "B" "E"; "C" "F"];
v = ["A" "B"  "C" "D"  "E" "F"];
test_matrix_typemaps("CharPtr", m, v, strcat(m), m + m, v + v);

m = [%T  %F;  %F  %T;  %T  %F];
v = [%T  %F   %T  %F   %T  %F];
test_matrix_typemaps("Bool", m, v, %T, ~m, ~v);

exec("swigtest.quit", -1);
