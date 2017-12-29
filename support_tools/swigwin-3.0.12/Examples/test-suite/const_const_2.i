/* This interface file tests whether SWIG handles doubly constant
   methods right. SF Bug #216057 against Swig 1.3a5, reported by
   Mike Romberg <romberg@users.sf.net>
*/

%module const_const_2

%inline %{
class Spam { 
public: 
  Spam() {} 
}; 

class Eggs { 
 public: 
 Eggs() {} 

 const Spam *spam(void) const { return new Spam(); } 
}; 

 %}
