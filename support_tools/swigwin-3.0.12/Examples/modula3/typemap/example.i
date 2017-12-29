/* File : example.i */
%module Example

%pragma(modula3) unsafe="true";

%insert(m3wrapintf) %{FROM ExampleRaw IMPORT Window, Point;
%}
%insert(m3wrapimpl) %{FROM ExampleRaw IMPORT Window, Point;
IMPORT M3toC;
IMPORT Ctypes AS C;
%}

/* Typemap applied to patterns of multiple arguments */

%typemap(m3rawinmode)   (char *outstr) %{VAR%}
%typemap(m3rawintype)   (char *outstr) %{CHAR%}
%typemap(m3wrapinmode)  (char *outstr, int size) %{VAR%}
%typemap(m3wrapintype)  (char *outstr, int size) %{ARRAY OF CHAR%}
%typemap(m3wrapargraw)  (char *outstr, int size) %{$1_name[0], NUMBER($1_name)%}


%typemap(m3rawinmode)   (const struct Window *) %{READONLY%}
%typemap(m3wrapinmode)  (const struct Window *) %{READONLY%}
%typemap(m3rawintype)   (      struct Window *) %{Window%}
%typemap(m3wrapintype)  (      struct Window *) %{Window%}

%typemap(m3rawinmode)   (const char *str []) %{READONLY%}
%typemap(m3wrapinmode)  (const char *str []) %{READONLY%}
%typemap(m3rawintype)   (const char *str []) %{(*ARRAY OF*) C.char_star%}
%typemap(m3wrapintype)  (const char *str []) %{ARRAY OF TEXT%}
%typemap(m3wrapargvar)  (const char *str []) %{$1: REF ARRAY OF C.char_star;%}
%typemap(m3wrapargraw)  (const char *str []) %{$1[0]%}
%typemap(m3wrapinconv)  (const char *str []) %{$1:= NEW(REF ARRAY OF C.char_star,NUMBER($1_name));
FOR i:=FIRST($1_name) TO LAST($1_name) DO
$1[i]:=M3toC.SharedTtoS($1_name[i]);
END;%}
%typemap(m3wrapfreearg) (const char *str [])
%{FOR i:=FIRST($1_name) TO LAST($1_name) DO
M3toC.FreeSharedS($1_name[i],$1[i]);
END;%}

%typemap(m3wraprettype) char * %{TEXT%}
%typemap(m3wrapretvar)  char * %{result_string: C.char_star;%}
%typemap(m3wrapretraw)  char * %{result_string%}
%typemap(m3wrapretconv) char * %{M3toC.CopyStoT(result_string)%}

struct Window {
  char *label;
  int left,top,width,height;
};


%typemap(m3wrapinname) (int x, int y) %{p%}
%typemap(m3wrapinmode) (int x, int y) %{READONLY%}
%typemap(m3wrapintype) (int x, int y) %{Point%}
%typemap(m3wrapargraw) (int x, int y) %{p.$1_name, p.$2_name%}

%typemap(m3wrapargraw)  (int &x, int &y) %{p.$1_name, p.$2_name%}
%typemap(m3wrapintype)  (int &x, int &y) %{Point%}
%typemap(m3wrapoutname) (int &x, int &y) %{p%}
%typemap(m3wrapouttype) (int &x, int &y) %{Point%}
%typemap(m3wrapargdir)  (int &x, int &y) "out"


%typemap(m3wrapargvar)  int &left, int &top, int &width, int &height "$1:C.int;"
%typemap(m3wrapargraw)  int &left, int &top, int &width, int &height "$1"
%typemap(m3wrapoutconv) int &left, int &top, int &width, int &height "$1"

%typemap(m3wrapargdir)  int &left, int &top "out"

%typemap(m3wrapouttype) int &width, int &height "CARDINAL"
%typemap(m3wrapargdir)  int &width, int &height "out"

struct Point {
  int x,y;
};

%m3multiretval get_box;

void  set_label       (      struct Window *win, const char *str, bool activate);
void  set_multi_label (      struct Window *win, const char *str []);
void  write_label     (const struct Window *win,       char *outstr, int size);
int   get_label       (const struct Window *win,       char *outstr, int size);
char *get_label_ptr   (const struct Window *win);
void  move(struct Window *win, int x, int y);
int   get_area(const struct Window *win);
void  get_box(const struct Window *win, int &left, int &top, int &width, int &height);
void  get_left(const struct Window *win, int &left);
void  get_mouse(const struct Window *win, int &x, int &y);
int   get_attached_data(const struct Window *win, const char *id);
