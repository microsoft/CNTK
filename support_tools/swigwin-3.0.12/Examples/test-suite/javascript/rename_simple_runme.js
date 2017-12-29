var rename_simple = require("rename_simple");
var NewStruct = rename_simple.NewStruct;

var s = new NewStruct();

// renamed instance variable
if (s.NewInstanceVariable !== 111) {
  throw new Error("NewInstanceVariable: Expected 111, was " + s.NewInstanceVariable);
}

// renamed instance method
if (s.NewInstanceMethod() !== 222) {
  throw new Error("NewInstanceMethod(): Expected 222, was " + s.NewInstanceMethod());
}

// renamed static method
if (NewStruct.NewStaticMethod() !== 333) {
  throw new Error("NewInstanceMethod(): Expected 333, was " + NewStruct.NewStaticMethod());
}

// renamed static variable
if (NewStruct.NewStaticVariable !== 444) {
  throw new Error("NewInstanceMethod(): Expected 444, was " + NewStruct.NewStaticVariable);
}

// renamed global function
if (rename_simple.NewFunction() !== 555) {
  throw new Error("rename_simple.NewFunction(): Expected 555, was " + rename_simple.NewFunction());
}

// renamed global variable
if (rename_simple.NewGlobalVariable !== 666) {
  throw new Error("rename_simple.NewGlobalVariable: Expected 666, was " + rename_simple.NewGlobalVariable);
}

// setting renamed variables
s.NewInstanceVariable = 1111;
if (s.NewInstanceVariable !== 1111) {
  throw new Error("NewInstanceVariable: Expected 1111, was " + s.NewInstanceVariable);
}

NewStruct.NewStaticVariable = 4444;
if (NewStruct.NewStaticVariable !== 4444) {
  throw new Error("NewInstanceMethod(): Expected 4444, was " + NewStruct.NewStaticVariable);
}

rename_simple.NewGlobalVariable = 6666;
if (rename_simple.NewGlobalVariable !== 6666) {
  throw new Error("rename_simple.NewGlobalVariable: Expected 6666, was " + rename_simple.NewGlobalVariable);
}
