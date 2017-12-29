%module template_expr

// bug #925555
%inline %{

  template<int __stride, class __elementTypeSequence,
    class __dataPtrType, class __elementType>
    inline const ThisType & 
    ConcatenationOf(const vctFixedLengthConstSequenceBase<_size - 1,
		    __stride, __elementTypeSequence, __dataPtrType> & other,
		    __elementType last);
%}

// bug #956282
%inline %{

  
template<int q>
class X {};


%}

%template(X_1) X<1>;
%template(X_m1) X<-1>;
%template(X_1plus2) X<1+2>;
%template(X_1024div8) X<1024/8>;


// bug #646275

%inline %{
    template<typename Type, short Rank>
	typedef typename CondRetType< Rank!=1,
		Test<Type,Rank-1>, Type>::TestRm1 TestRm1;
%}


// bug #994301


%inline %{
    template<int foo> class X994301 {};
%}

%template(X994301_ternary) X<(7 >= 4) ? 1 + 1 : 1>;
%template(X994301_ternary2) X<(7 <= 4) ? 1 + 1 : 1>;


// bug #1338527 (still broken)


%inline %{

  template<typename, typename> class PatchList;

%}
