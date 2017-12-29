%module template_default_pointer

%inline %{

template <class T1, class T2 = T1*>
class B
{
};

%}

%template(B_d) B<double>;
