// test of std::stack
%module li_std_stack

%include std_stack.i


%template( IntStack ) std::stack< int >;
