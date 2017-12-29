clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("ignore_parameter", .Platform$dynlib.ext, sep=""))
source("ignore_parameter.R")
cacheMetaData(1)

unittest(jaguar(1, 1.0), "hello")
q(save="no")



