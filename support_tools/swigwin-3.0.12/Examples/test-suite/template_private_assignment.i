%module template_private_assignment

/*
swig-devel mailing list report problem explained 2014-01-07
A setter for the global variable deleted_bits is generated because there is no template
instantiation for the template and hence SWIG does not find the private assignment operator.
SwigValueWrapper is probably on by default for templates that are not instantiated for the
same reason.
The solution is probably to add an instantiation of the template as soon as one is parsed,
that is an implicit empty %template().
*/

%inline %{
template<typename T, typename U> struct DeletedBits {
//  DeletedBits& operator=(const DeletedBits&) = delete;
private:
  DeletedBits& operator=(const DeletedBits&);
};

DeletedBits<int, double> deleted_bits;
%}

// This works around the problem
//%template() DeletedBits<int, double>;
