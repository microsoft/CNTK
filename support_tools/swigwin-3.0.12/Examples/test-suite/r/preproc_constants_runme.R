clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("preproc_constants", .Platform$dynlib.ext, sep=""))
source("preproc_constants.R")
cacheMetaData(1)

v <- enumToInteger('kValue', '_MyEnum')
print(v)
# temporarily removed until fixed (in progress, see Github patch #500)
#unittest(v,4)
q(save="no")
