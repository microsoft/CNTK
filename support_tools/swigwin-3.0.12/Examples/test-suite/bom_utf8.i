%module bom_utf8

/* Test for UTF8 BOM at start of file */
%inline %{
struct NotALotHere {
  int n;
};
%}

