clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))
dyn.load(paste("unions", .Platform$dynlib.ext, sep=""))
source("unions.R")
cacheMetaData(1)

ss <- SmallStruct()

bstruct <- BigStruct()

q(save="no")


