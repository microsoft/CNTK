%module li_constraints
%include <constraints.i>

%inline %{
void test_nonnegative(double NONNEGATIVE) {
}

void test_nonpositive(double NONPOSITIVE) {
}

void test_positive(double POSITIVE) {
}

void test_negative(double POSITIVE) {
}

void test_nonzero(double NONZERO) {
}

void test_nonnull(void *NONNULL) {
}

/* These generated non-portable code and there isn't an obvious fix

void test_align8(void *ALIGN8) {
}

void test_align4(void *ALIGN4) {
}

void test_align2(void *ALIGN2) {
}
*/
%}

