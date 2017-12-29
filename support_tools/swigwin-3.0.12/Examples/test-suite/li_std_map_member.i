%module li_std_map_member

%inline %{
int i;
class TestA {
public:
TestA() { i = 1; }
int i;
};
%}
%include std_pair.i
%include std_map.i

namespace std
{
%template(pairii) pair<int,int>;
%template(mapii) map<int,int>;
%template(pairita) pair<int,TestA>;
%template(mapita) map<int,TestA>;
}
