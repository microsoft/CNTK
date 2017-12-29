
if [ catch { load ./overload_simple[info sharedlibextension] overload_simple} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

set f [new_Foo]
set b [new_Bar]
set v [malloc_void 32]

set x [foo 3]
if {$x != "foo:int"} {
   puts stderr "foo(int) test failed $x"
   exit 1
}

set x [foo 3.4]
if {$x != "foo:double"} {
   puts stderr "foo(double) test failed"
   exit 1
}

set x [foo hello]
if {$x != "foo:char *"} {
   puts stderr "foo(char *) test failed"
   exit 1
}

set x [foo $f]
if {$x != "foo:Foo *"} {
   puts stderr "foo(Foo *) test failed"
   exit 1
}

set x [foo $b]
if {$x != "foo:Bar *"} {
   puts stderr "foo(Bar *) test failed"
   exit 1
}

set x [foo $v]
if {$x != "foo:void *"} {
   puts stderr "foo(void *) test failed"
   exit 1
}

Spam s

set x [s foo 3]
if {$x != "foo:int"} {
   puts stderr "Spam::foo(int) test failed"
   exit 1
}

set x [s foo 3.4]
if {$x != "foo:double"} {
   puts stderr "Spam::foo(double) test failed"
   exit 1
}

set x [s foo hello]
if {$x != "foo:char *"} {
   puts stderr "Spam::foo(char *) test failed"
   exit 1
}

set x [s foo $f]
if {$x != "foo:Foo *"} {
   puts stderr "Spam::foo(Foo *) test failed"
   exit 1
}

set x [s foo $b]
if {$x != "foo:Bar *"} {
   puts stderr "Spam::foo(Bar *) test failed"
   exit 1
}

set x [s foo $v]
if {$x != "foo:void *"} {
   puts stderr "Spam::foo(void *) test failed"
   exit 1
}


set x [Spam_bar 3]
if {$x != "bar:int"} {
   puts stderr "Spam::bar(int) test failed"
   exit 1
}

set x [Spam_bar 3.4]
if {$x != "bar:double"} {
   puts stderr "Spam::bar(double) test failed"
   exit 1
}

set x [Spam_bar hello]
if {$x != "bar:char *"} {
   puts stderr "Spam::bar(char *) test failed"
   exit 1
}

set x [Spam_bar $f]
if {$x != "bar:Foo *"} {
   puts stderr "Spam::bar(Foo *) test failed"
   exit 1
}

set x [Spam_bar $b]
if {$x != "bar:Bar *"} {
   puts stderr "Spam::bar(Bar *) test failed"
   exit 1
}

set x [Spam_bar $v]
if {$x != "bar:void *"} {
   puts stderr "Spam::bar(void *) test failed"
   exit 1
}

Spam s
set x [s cget -type]
if {$x != "none"} {
    puts stderr "Spam() test failed"
}

Spam s 3
set x [s cget -type]
if {$x != "int"} {
    puts stderr "Spam(int) test failed"
}

Spam s 3.4
set x [s cget -type]
if {$x != "double"} {
    puts stderr "Spam(double) test failed"
}

Spam s hello
set x [s cget -type]
if {$x != "char *"} {
    puts stderr "Spam(char *) test failed"
}

Spam s $f
set x [s cget -type]
if {$x != "Foo *"} {
    puts stderr "Spam(Foo *) test failed"
}

Spam s $b
set x [s cget -type]
if {$x != "Bar *"} {
    puts stderr "Spam(Bar *) test failed"
}

Spam s $v
set x [s cget -type]
if {$x != "void *"} {
    puts stderr "Spam(void *) test failed"
}

free_void $v



