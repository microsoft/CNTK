%module r_overload_comma

%inline %{
class r_overload_comma
{
  public:
  int getMember1()const {return _member1;}
  void setMember1ThatEndsWithWord_get(int arg) { _member1=arg; }
  void setMember1ThatEndsWithWord_get(char* arg) {_member1=atoi(arg);}
  
  private:
  int _member1;
};
 %}
