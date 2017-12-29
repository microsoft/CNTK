
if [ catch { load ./import_nomodule[info sharedlibextension] import_nomodule} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}
