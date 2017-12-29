
// This testcase tests the JNI types

%module java_jnitypes

%inline %{

jboolean      jnifunc_bool(jboolean in) { return in; } /* some JVM implementations won't allow overloading of the jboolean type with some of the others on the c++ level */
jchar         jnifunc(jchar in) { return in; }
jbyte         jnifunc(jbyte in) { return in; }
jshort        jnifunc(jshort in) { return in; }
jint          jnifunc(jint in) { return in; }
jlong         jnifunc(jlong in) { return in; }
jfloat        jnifunc(jfloat in) { return in; }
jdouble       jnifunc(jdouble in) { return in; }
jstring       jnifunc(jstring in) { return in; }
jobject       jnifunc(jobject in) { return in; }
jbooleanArray jnifunc(jbooleanArray in) { return in; }
jcharArray    jnifunc(jcharArray in) { return in; }
jbyteArray    jnifunc(jbyteArray in) { return in; }
jshortArray   jnifunc(jshortArray in) { return in; }
jintArray     jnifunc(jintArray in) { return in; }
jlongArray    jnifunc(jlongArray in) { return in; }
jfloatArray   jnifunc(jfloatArray in) { return in; }
jdoubleArray  jnifunc(jdoubleArray in) { return in; }
jobjectArray  jnifunc(jobjectArray in) { return in; }

%}

