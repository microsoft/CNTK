/* This is a test case for -*- C -*- mode. */
%module tag_no_clash_with_variable

%inline %{

/* error_action is only a tag, not a type... */
enum error_action {
    ERRACT_ABORT,
    ERRACT_EXIT, 
    ERRACT_THROW
};

/* ... thus it does not clash with a variable of the same name. */ 
enum error_action error_action;

/* Likewise for structs: */

struct buffalo { 
  int foo;
};

struct buffalo buffalo; 

/* And for union */

union onion {
  int cheese;
};

union onion onion;
 
%}

