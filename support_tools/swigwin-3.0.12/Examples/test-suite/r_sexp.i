%module r_sexp

extern "C" SEXP return_sexp(SEXP x);

%inline %{
SEXP return_sexp(SEXP x) {
  return x; //Rcpp NumericVector is automatically casted to SEXP
}
%}

