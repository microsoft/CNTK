// File: runme.js

// This file illustrates class C++ interface generated
// by SWIG.

var example = require("example");

// Calling a module function ( aka global function )
if (example.module_function() !== 7) {
  throw new Error("example.module_function() should equal 7");
}
console.log("example.module_function(): " + example.module_function());

// Accessing a module (aka global) variable
if (example.module_variable !== 9) {
  throw new Error("example.module_variable should equal 9");
}
console.log("example.module_variable: " + example.module_variable);

// Creating an instance of the class
var w1 = new example.MyWorld.World();
console.log("Creating class instance: w1 = new example.MyWorld.World(): " + w1);

// Accessing class members
if (w1.create_world() !== 17) {
  throw new Error("w1.create_world() should equal 17");
}
console.log("w1.create_world() = " + w1.create_world());

if (w1.world_max_count !== 9) {
  throw new Error("w1.world_max_count should equal 9");
}
console.log("w1.world_max_count = " + w1.world_max_count);

// Accessing enums from class within namespace
if (example.MyWorld.Nested.Dweller.MALE !== 0) {
  throw new Error("example.MyWorld.Nested.Dweller.MALE should equal 0");
}
console.log("Accessing enums: ex.MyWorld.Nested.Dweller.MALE = " + example.MyWorld.Nested.Dweller.MALE);

if (example.MyWorld.Nested.Dweller.FEMALE !== 1) {
  throw new Error("example.MyWorld.Nested.Dweller.FEMALE should equal 1");
}
console.log("Accessing enums: ex.MyWorld.Nested.Dweller.FEMALE = " + example.MyWorld.Nested.Dweller.FEMALE);

// Accessing static member function
if (example.MyWorld.Nested.Dweller.count() !== 19) {
  throw new Error("example.MyWorld.Nested.Dweller.count() should equal 19");
}
console.log("Accessing static member function: ex.MyWorld.Nested.Dweller.count() = " + example.MyWorld.Nested.Dweller.count());
