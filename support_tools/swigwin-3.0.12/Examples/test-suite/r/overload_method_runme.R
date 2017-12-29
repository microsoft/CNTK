clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("overload_method", .Platform$dynlib.ext, sep=""))
source("overload_method.R")
cacheMetaData(1)

b <- Base()
Base_method(b)
Base_overloaded_method(b)
Base_overloaded_method(b, 43)
Base_overloaded_method(b)
b$method()

b$overloaded_method()
