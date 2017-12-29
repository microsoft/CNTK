
if [ catch { load ./overload_copy[info sharedlibextension] overload_copy} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

Foo f
Foo g [f cget -this]





