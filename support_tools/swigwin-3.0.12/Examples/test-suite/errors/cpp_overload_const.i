%module xxx

void check(const int *v) {}
void check(int *v) {}
void check(int &v) {}
void check(const int &v) {} // note: no warning as marshalled by value

struct OverStruct {};
void check(const OverStruct *v) {}
void check(OverStruct *v) {}
void check(OverStruct &v) {}
void check(const OverStruct &v) {}

