%module ruby_manual_proxy


%typemap(in, numinputs=0) SWIGTYPE ** ($*1_ltype temp) "$1 = &temp;";

%typemap(argout) SWIGTYPE **OUTPARAM {
  $result = SWIG_Ruby_AppendOutput($result, SWIG_NewPointerObj(*$1, $*1_descriptor, 0));
}

%apply SWIGTYPE **OUTPARAM {
  svn_fs_t **
};

%typemap(check) svn_fs_t * {
  if (!$1) {
    svn_swig_rb_raise_svn_fs_already_close();
  }
}

%{
typedef struct svn_fs_t {
  char path[256];
} svn_fs_t;

void svn_fs_create(svn_fs_t **fs_p, const char *path) {
  svn_fs_t *fs = (svn_fs_t *)malloc(sizeof(svn_fs_t));
  strncpy(fs->path, path, 256);
  *fs_p = fs;
}
const char *svn_fs_path(svn_fs_t *fs) {
  return fs->path;
}
%}

typedef struct svn_fs_t svn_fs_t;
void svn_fs_create(svn_fs_t **fs_p, const char *path);
const char *svn_fs_path(svn_fs_t *fs);

%{
static void svn_swig_rb_raise_svn_fs_already_close(void) {
  rb_raise(rb_eIOError, "already closed");
}

static VALUE svn_fs_swig_rb_close(VALUE self) {
  if (!DATA_PTR(self)) {
    svn_swig_rb_raise_svn_fs_already_close();
  }

  DATA_PTR(self) = NULL;

  return Qnil;
}

static VALUE svn_fs_swig_rb_closed(VALUE self) {
  return DATA_PTR(self) ? Qfalse : Qtrue;
}
%}

%insert("init") %{
  {
    VALUE cSvnfs;
    cSvnfs = rb_const_get(_mSWIG, rb_intern("TYPE_p_svn_fs_t"));
    rb_define_method(cSvnfs, "close",
                     VALUEFUNC(svn_fs_swig_rb_close), 0);
  }
%}
