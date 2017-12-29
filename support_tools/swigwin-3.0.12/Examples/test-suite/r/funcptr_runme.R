clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("funcptr", .Platform$dynlib.ext, sep=""))
source("funcptr.R")
cacheMetaData(1)
unittest(do_op(1, 3, add), 4)
unittest(do_op(2, 3, multiply), 6)
unittest(do_op(2, 3, funcvar()), 5)
