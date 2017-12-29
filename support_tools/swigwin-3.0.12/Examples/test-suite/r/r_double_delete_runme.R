dyn.load(paste("r_double_delete", .Platform$dynlib.ext, sep=""))
source("r_double_delete.R")
cacheMetaData(1)

# ----- Object creation -----

f <- Foo(2.0)
delete(f);
delete(f);
