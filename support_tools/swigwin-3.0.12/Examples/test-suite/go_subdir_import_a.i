/* File : go_subdir_import_a.i */

/*
 * This files helps check the case where the SWIG-generated .go file needs to
 * import another, SWIG-generated, module that is in a relative subdirectory.
 * This case might happen for two different reasons:
 * 1) Importing a module for which the .i file is in a subdirectory relatively
 *    to this file (this is tested here with go_subdir_import_c).
 * 2) Importing a module whos module name is a path (this is tested here with
 *    go_subdir_import_b).
 *
 * This file is the "root" file that imports the two modules which will be
 * generated (by swig) in a relative subdirectory.
 */
%module go_subdir_import_a

%import(module="testdir/go_subdir_import/go_subdir_import_c") "testdir/go_subdir_import/go_subdir_import_c.i"
%import "go_subdir_import_b.i"

%{
class ObjC {
 public:
  virtual int getInt() const;
};

class ObjB {
 public:
  virtual int getInt() const;
};
%}

%inline %{
int AddFive(const ObjB& b, const ObjC& c) {
  return b.getInt() + c.getInt() + 5;
}
%}

