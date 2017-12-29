// A function that passes arrays by reference 

%module arrayref

%inline %{

void foo(const int (&x)[10]) {
}

void bar(int (&x)[10]) {
}
%}


