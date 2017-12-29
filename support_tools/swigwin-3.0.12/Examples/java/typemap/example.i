/* File : example.i */
%module example
%{
/*
   example of a function that returns a value in the char * argument
   normally used like:

   char buf[bigenough];
   f1(buf);
*/

void f1(char *s) {
  if(s != NULL) {
    sprintf(s, "hello world");
  }
}

void f2(char *s) {
  f1(s);
}

void f3(char *s) {
  f1(s);
}

%}

/* default behaviour is that of input arg, Java cannot return a value in a 
 * string argument, so any changes made by f1(char*) will not be seen in the Java
 * string passed to the f1 function.
*/
void f1(char *s);

%include various.i

/* use the BYTE argout typemap to get around this. Changes in the string by 
 * f2 can be seen in Java. */
void f2(char *BYTE);



/* Alternative approach uses a StringBuffer typemap for argout */

/* Define the types to use in the generated JNI C code and Java code */
%typemap(jni) char *SBUF "jobject"
%typemap(jtype) char *SBUF "StringBuffer"
%typemap(jstype) char *SBUF "StringBuffer"

/* How to convert Java(JNI) type to requested C type */
%typemap(in) char *SBUF {

  $1 = NULL;
  if($input != NULL) {
    /* Get the String from the StringBuffer */
    jmethodID setLengthID;
    jclass sbufClass = (*jenv)->GetObjectClass(jenv, $input);
    jmethodID toStringID = (*jenv)->GetMethodID(jenv, sbufClass, "toString", "()Ljava/lang/String;");
    jstring js = (jstring) (*jenv)->CallObjectMethod(jenv, $input, toStringID);

    /* Convert the String to a C string */
    const char *pCharStr = (*jenv)->GetStringUTFChars(jenv, js, 0);

    /* Take a copy of the C string as the typemap is for a non const C string */
    jmethodID capacityID = (*jenv)->GetMethodID(jenv, sbufClass, "capacity", "()I");
    jint capacity = (*jenv)->CallIntMethod(jenv, $input, capacityID);
    $1 = (char *) malloc(capacity+1);
    strcpy($1, pCharStr);

    /* Release the UTF string we obtained with GetStringUTFChars */
    (*jenv)->ReleaseStringUTFChars(jenv,  js, pCharStr);

    /* Zero the original StringBuffer, so we can replace it with the result */
    setLengthID = (*jenv)->GetMethodID(jenv, sbufClass, "setLength", "(I)V");
    (*jenv)->CallVoidMethod(jenv, $input, setLengthID, (jint) 0);
  }
}

/* How to convert the C type to the Java(JNI) type */
%typemap(argout) char *SBUF {

  if($1 != NULL) {
    /* Append the result to the empty StringBuffer */
    jstring newString = (*jenv)->NewStringUTF(jenv, $1);
    jclass sbufClass = (*jenv)->GetObjectClass(jenv, $input);
    jmethodID appendStringID = (*jenv)->GetMethodID(jenv, sbufClass, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;");
    (*jenv)->CallObjectMethod(jenv, $input, appendStringID, newString);

    /* Clean up the string object, no longer needed */
    free($1);
    $1 = NULL;
  }  
}
/* Prevent the default freearg typemap from being used */
%typemap(freearg) char *SBUF ""

/* Convert the jstype to jtype typemap type */
%typemap(javain) char *SBUF "$javainput"

/* apply the new typemap to our function */
void f3(char *SBUF);

