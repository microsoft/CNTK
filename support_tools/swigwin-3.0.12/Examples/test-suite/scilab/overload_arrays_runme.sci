exec("swigtest.start", -1);

// Functions

checkequal(foo(int32([1, 2, 3])), "foo:int[SIZE]", "foo(int[SIZE])");
checkequal(foo([1, 2, 3]), "foo:double[SIZE]", "foo(double[SIZE])");
checkequal(foo(["1" "2" "3"]), "foo:char *[SIZE]", "foo(char *[SIZE])");

// Class methods

s = new_Spam();
checkequal(Spam_foo(s, int32([1, 2, 3])), "foo:int[SIZE]", "Spam::foo(int[SIZE])");
checkequal(Spam_foo(s, [1, 2, 3]), "foo:double[SIZE]", "Spam::foo(double[SIZE])");
checkequal(Spam_foo(s, ["1" "2" "3"]), "foo:char *[SIZE]", "Spam::foo(char *[SIZE])");
delete_Spam(s);

// Static class methods

checkequal(Spam_bar(int32([1, 2, 3])), "bar:int[SIZE]", "Spam::bar(int[SIZE])");
checkequal(Spam_bar([1, 2, 3]), "bar:double[SIZE]", "Spam::bar(double[SIZE])");
checkequal(Spam_bar("hello"), "bar:char *[SIZE]", "Spam::bar(char *[SIZE])");

// Constructors

s = new_Spam();
checkequal(Spam_type_get(s), "none", "Spam::Spam()");
delete_Spam(s);

s = new_Spam(int32([1, 2, 3]));
checkequal(Spam_type_get(s), "int[SIZE]", "Spam::Spam(int[SIZE])");
delete_Spam(s);

s = new_Spam([1, 2, 3]);
checkequal(Spam_type_get(s), "double[SIZE]", "Spam::Spam(double[SIZE])");
delete_Spam(s);

s = new_Spam(["1" "2" "3"]);
checkequal(Spam_type_get(s), "char *[SIZE]", "Spam::Spam(char *[SIZE])");
delete_Spam(s);


a = new_ClassA();
b = ClassA_method1(a, [1 2 3]);


exec("swigtest.quit", -1);
