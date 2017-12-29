%module template_typedef_fnc

%include "std_vector.i"
namespace std {
   %template(IntVector) vector<int>;
};

%inline 
{
  typedef void (*RtMidiCallback)(std::vector<int> *message);

  void setCallback( RtMidiCallback callback) {
  }
}
