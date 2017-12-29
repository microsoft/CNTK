clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("integers", .Platform$dynlib.ext, sep=""))
source("integers.R")
cacheMetaData(1)

unittest(signed_char_identity(1), 1)
unittest(unsigned_char_identity(1), 1)
unittest(signed_short_identity(1), 1)
unittest(unsigned_short_identity(1), 1)
unittest(signed_int_identity(1), 1)
unittest(unsigned_int_identity(1), 1)
unittest(signed_long_identity(1), 1)
unittest(unsigned_long_identity(1), 1)
unittest(signed_long_long_identity(1), 1)
unittest(unsigned_long_long_identity(1), 1)

q(save="no")



