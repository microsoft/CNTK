clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("r_legacy", .Platform$dynlib.ext, sep=""))
source("r_legacy.R")
cacheMetaData(1)

obj <- getObject(1,3)
unittest(class(obj), "_p_Obj")
unittest(obj$i, 1)
unittesttol(obj$d, 3, 0.001)
unittest(obj$str, "a test string")
obj$i <- 2
unittest(obj$i, 2)
obj$d <- 4
unittesttol(obj$d, 4, 0.001)
obj$str <- "a new string"
unittest(obj$str, "a new string")

unittest(getInt(), 42)
unittesttol(getDouble(),3.14159, 0.001)
unittesttol(getFloat(),3.14159/2.0, 0.001)
unittest(getLong(), -321313)  
unittest(getUnsignedLong(), 23123)
unittest(getChar(), "A")

q(save="no")





