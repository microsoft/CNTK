clargs <- commandArgs(trailing=TRUE)
source(file.path(clargs[1], "unittest.R"))

dyn.load(paste("arrays_dimensionless", .Platform$dynlib.ext, sep=""))
source("arrays_dimensionless.R")
cacheMetaData(1)

unittest(arr_short(1:4, 3), 6)
unittest(arr_ushort(1:4, 3), 6)
unittest(arr_int(1:4, 3), 6)
unittest(arr_uint(1:4, 3), 6)
unittest(arr_long(1:4, 3), 6)
unittest(arr_ulong(1:4, 3), 6)
unittest(arr_ll(1:4, 3), 6)
unittest(arr_ull(1:4, 3), 6)
unittest(arr_float(as.numeric(1:4), 3), 6)
unittest(arr_double(as.numeric(1:4), 3), 6)

q(save="no")



