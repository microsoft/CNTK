clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))
dyn.load(paste("simple_array", .Platform$dynlib.ext, sep=""))
source("simple_array.R")
cacheMetaData(1)
initArray()

q(save="no")


