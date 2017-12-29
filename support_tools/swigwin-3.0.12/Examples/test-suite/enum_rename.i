%module enum_rename

%warnfilter(SWIGWARN_PARSE_REDEFINED) S_May;

// %rename using regex can do the equivalent of these two renames, which was resulting in uncompilable code
%rename(May) M_May;
%rename(May) S_May;

%inline %{
	enum Month { M_Jan, M_May, M_Dec };
	enum Severity { S_May, S_Can, S_Must };
%}
