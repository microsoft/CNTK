clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("rename_simple", .Platform$dynlib.ext, sep=""))
source("rename_simple.R")
cacheMetaData(1)

s <- NewStruct();
unittest(111, s$NewInstanceVariable)
unittest(222, s$NewInstanceMethod())
unittest(333, NewStruct_NewStaticMethod())
unittest(444, NewStruct_NewStaticVariable())
unittest(555, NewFunction())
unittest(666, NewGlobalVariable())

s$NewInstanceVariable <- 1111
NewStruct_NewStaticVariable(4444)
NewGlobalVariable(6666)

unittest(1111, s$NewInstanceVariable)
unittest(4444, NewStruct_NewStaticVariable())
unittest(6666, NewGlobalVariable())
