%module li_std_pair_using

%include<stl.i>
using std::pair;

%template(StringStringPair) pair<std::string, std::string>;

%inline %{
typedef int Integer;
using std::string;
%}

%template(StringIntPair) pair<string, int>;

%inline %{
typedef std::string String;
typedef string Streeng;
std::pair<String, Streeng> bounce(std::pair<std::string, string> p) {
  return p;
}
%}
