%module python_pybuf
%include<pybuffer.i>
%include<cstring.i>
/*functions for the test case*/
%pybuffer_mutable_binary(char *buf1, int len);
%pybuffer_mutable_string(char *buf2);
%pybuffer_binary(const char *buf3, int len);
%pybuffer_string(const char *buf4);

%inline %{
  void func1(char *buf1, int len)
  {
    int i;
    for (i=0; i<len; ++i)
      buf1[i] = 'a';
    return;
  }
  void func2(char *buf2)
  {
    strcpy(buf2, "Hello world!");
  }
  int func3(const char *buf3, int len)
  {
    int count = 0;
    int i;
    for(i=0; i<len; ++i)
      if (isalnum(buf3[i]))
        ++count;
    return count;
  }
  int func4(const char *buf4)
  {
    return strlen(buf4);
  }  
%}

/*functions for the benchmark*/
%pybuffer_mutable_string(char *str1);
%cstring_mutable(char *str2);

%inline %{
void title(char *str) {
    int outword = 1;
    while(*str) {
        if (isalnum(*str)) {
            if (outword) {
                outword = 0;
                *str = toupper(*str);
            }            
        }
        else {
            outword = 0;
        }
        str++;
    }
}

void title1(char *str1) {
    title(str1);
}
void title2(char *str2) {
    title(str2);
}
%}
