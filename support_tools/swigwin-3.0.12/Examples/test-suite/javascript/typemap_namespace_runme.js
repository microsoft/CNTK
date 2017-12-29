var typemap_namespace = require("typemap_namespace");

if (typemap_namespace.test1("hello") != "hello")
    throw "RuntimeError";

if (typemap_namespace.test2("hello") != "hello")
    throw "RuntimeError";
