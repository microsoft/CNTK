// test STL containers typemaps

exec("swigtest.start", -1);

function checkerror(ierr, cmd)
  if ierr <> 0 then swigtesterror("error " + string(ierr) + " in """ + cmd + """"); end
endfunction

// test container of pointers returned from fonction (expected a list)
function [classAPtr_list, classAPtr1, classAPtr2] = testCreateContainerPtr(container, value1, value2)
  classAPtr1 = new_ClassA(value1);
  classAPtr2 = new_ClassA(value2);
  func = msprintf("ret_ClassAPtr_%s", container);
  cmd = msprintf("classAPtr_list = %s(classAPtr1, classAPtr2);", func);
  ierr = execstr(cmd, "errcatch");
  if ierr <> 0 then swigtesterror("error in " + cmd); end
  if ~exists('classAPtr_list') | (size(classAPtr_list) <> 2) then
    swigtesterror(func);
  end

  checkequal(ClassA_a_get(classAPtr_list(1)), value1, "ClassA_a_get(classAPtr_list(1))");
  checkequal(ClassA_a_get(classAPtr_list(2)), value2, "ClassA_a_get(classAPtr_list(2))");
endfunction

// test a given container of pointer
// -container: type of container: "vector", "set"...
// -value1, value2: values to store in container
// -expected_accumulate_value: expected value of an accumulation function
//    computed on the container
function testContainerPtr(container, value1, value2, expected_accumulate_value)
  // test container of pointers returned from flonction (expected a list)
  [classAPtr_list, classAPtr1, classAPtr2] = testCreateContainerPtr(container, value1, value2);

  // test container passed as value of function
  func = msprintf("val_ClassAPtr_%s", container);
  cmd = msprintf("classAPtr = %s(classAPtr_list);", func);
  ierr = execstr(cmd, "errcatch");
  checkerror(ierr, cmd);
  checkequal(ClassA_a_get(classAPtr), expected_accumulate_value, func);

  // recreate a container
  [classAPtr_list, classAPtr1, classAPtr2] = testCreateContainerPtr(container, value1, value2);

  // test container passed as reference of function
  func = msprintf("ref_ClassAPtr_%s", container);
  cmd = msprintf("classAPtr = %s(classAPtr_list);", func);
  ierr = execstr(cmd, "errcatch");
  checkerror(ierr, cmd);
  checkequal(ClassA_a_get(classAPtr), expected_accumulate_value, func);
endfunction

// test a given container of a given primitive type
// -container: type of container: "vector", "set"...
// -value_type: type of element stored in container: "int", ...
// -value1, value2: values to store in container
// -expected_accumulate_value: expected value of an accumulation function
//     computed on the container
function testContainerType(container, value_type, value1, value2, ..
  expected_returned_container, expected_accumulate_value)
  // test container of basic type returned from fonction
  func = msprintf("ret_%s_%s", value_type, container);
  if value_type == "string" then
    cmd = msprintf("c = %s(''%s'', ''%s'');", func, value1, value2);
  elseif value_type == "bool" then
    cmd = msprintf("c = %s(%s, %s);", func, "%"+string(value1), "%"+string(value2));
  else
    cmd = msprintf("c = %s(%d, %d);", func, value1, value2);
  end
  ierr = execstr(cmd, "errcatch");
  checkerror(ierr, cmd);
  checkequal(c, expected_returned_container, func);

  // test container passed as value of function
  func = msprintf("val_%s_%s", value_type, container);
  cmd = msprintf("s = %s(c);", func);
  ierr = execstr(cmd, "errcatch");
  checkerror(ierr, cmd);
  checkequal(s, expected_accumulate_value, func);

  // test container passed as reference of function
  func = msprintf("ref_%s_%s", value_type, container);
  cmd = msprintf("s = %s(c);", func);
  ierr = execstr(cmd, "errcatch");
  checkerror(ierr, cmd);
  checkequal(s, expected_accumulate_value, func);
endfunction

// test a given container of different types
// -container: type of container: "vector", "set"...
function testContainer(container)
  testContainerType(container, "int", 1, 2, [1, 2], 3);
  testContainerType(container, "double", 2., 3., [2., 3.], 5.);
  testContainerType(container, "float", 2., 3., [2., 3.], 5.);
  testContainerType(container, "string", "a", "b", ["a", "b"], "ab");
  testContainerType(container, "bool", %F, %T, [%F, %T], %T);
  testContainerPtr("vector", 1, 3, 4);
endfunction


testContainer("vector");
testContainer("list");
testContainer("deque");
testContainer("set");
testContainer("multiset");

exec("swigtest.quit", -1);


