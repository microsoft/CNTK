%module example

%typemap(in) (int nattributes, const char **names, const int *values) (VALUE keys_ary, int i, VALUE key, VALUE val) {
  Check_Type($input, T_HASH);
  $1 = NUM2INT(rb_funcall($input, rb_intern("size"), 0, NULL));
  $2 = NULL;
  $3 = NULL;
  if ($1 > 0) {
    $2 = (char **) malloc($1*sizeof(char *));
    $3 = (int *) malloc($1*sizeof(int));
    keys_ary = rb_funcall($input, rb_intern("keys"), 0, NULL);
    for (i = 0; i < $1; i++) {
      key = rb_ary_entry(keys_ary, i);
      val = rb_hash_aref($input, key);
      Check_Type(key, T_STRING);
      Check_Type(val, T_FIXNUM);
      $2[i] = StringValuePtr(key);
      $3[i] = NUM2INT(val);
    }
  }
}

%typemap(freearg) (int nattributes, const char **names, const int *values) {
  free((void *) $2);
  free((void *) $3);
}

%inline %{
void setVitalStats(const char *person, int nattributes, const char **names, const int *values) {
  int i;
  printf("Name: %s\n", person);
  for (i = 0; i < nattributes; i++) {
    printf("  %s => %d\n", names[i], values[i]);
  }
}
%}
