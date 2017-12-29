
if [ catch { load ./li_std_string[info sharedlibextension] li_std_string} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}


Structure s 
if {"[s cget -MemberString2]" != "member string 2"} { error "bad string map"}
s configure -MemberString2 "hello"
if {"[s cget -MemberString2]" != "hello"} { error "bad string map"}

if {"[s cget -ConstMemberString]" != "const member string"} { error "bad string map"}

if {"$GlobalString2" != "global string 2"} { error "bad string map"}
if {"$Structure_StaticMemberString2" != "static member string 2"} { error "bad string map"}

set GlobalString2 "hello"
if {"$GlobalString2" != "hello"} { error "bad string map"}

set Structure_StaticMemberString2 "hello"
if {"$Structure_StaticMemberString2" != "hello"} { error "bad string map"}
