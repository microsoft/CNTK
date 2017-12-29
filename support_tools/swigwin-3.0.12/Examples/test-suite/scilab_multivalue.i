%module scilab_multivalue



void output2(int *OUTPUT, int *OUTPUT);
int output2Ret(int *OUTPUT, int *OUTPUT);
void output2Input2(int a, int b, int *OUTPUT, int *OUTPUT);
int output2Input2Ret(int a, int b, int *OUTPUT, int *OUTPUT);
int output3Input1Ret(int a, int *OUTPUT, int *OUTPUT, int *OUTPUT);
int output3Input3Ret(int x, int *OUTPUT, int y, int *OUTPUT, int z, int *OUTPUT);

void inout2(int *INOUT, int *INOUT);
int inout2Ret(int *INOUT, int *INOUT);
void inout2Input2(int a, int b, int *INOUT, int *INOUT);
int inout2Input2Ret(int a, int b, int *INOUT, int *INOUT);
int inout3Input1Ret(int a, int *INOUT, int *INOUT, int *INOUT);
int inout3Input3Ret(int x, int *INOUT, int y, int *INOUT, int z, int *INOUT);

class ClassA {
public:
  ClassA() {};
  int output2Input2Ret(int a, int b, int *OUTPUT, int *OUTPUT);
  int inout2Input2Ret(int a, int b, int *INOUT, int *INOUT);
};

%{

// Test return of multiple values with OUTPUT

void output2(int *a, int *b) {
  *a = 1;
  *b = 2;
}

int output2Ret(int *a, int *b) {
  *a = 1;
  *b = 2;
  return *a + *b;
}

void output2Input2(int a, int b, int *c, int *d) {
  *c = a + 1;
  *d = b + 2;
}

int output2Input2Ret(int a, int b, int *c, int *d) {
  *c = a + 1;
  *d = b + 2;
  return *c + *d;
}

int output3Input1Ret(int x, int *a, int *b, int *c) {
  *a = x + 1;
  *b = x + 2;
  *c = x + 3;
  return x;
}

int output3Input3Ret(int x, int *a, int y, int *b, int z, int *c) {
  *a = x + 1;
  *b = y + 2;
  *c = z + 3;
  return *a + *b + *c;
}


// Test return of multiple values with INOUT

void inout2(int *a, int *b) {
  *a = *a + 1;
  *b = *a + 2;
}

int inout2Ret(int *a, int *b) {
  *a = *a + 1;
  *b = *a + 2;
  return *a + *b;
}

void inout2Input2(int a, int b, int *c, int *d) {
  *c = *c + a;
  *d = *d + b;
}

int inout2Input2Ret(int a, int b, int *c, int *d) {
  *c = *c + a;
  *d = *d + b;
  return *c + *d;
}

int inout3Input1Ret(int x, int *a, int *b, int *c) {
  *a = *a + x;
  *b = *b + x;
  *c = *c + x;
  return x;
}

int inout3Input3Ret(int x, int *a, int y, int *b, int z, int *c) {
  *a = *a + x;
  *b = *b + y;
  *c = *c + z;
  return *a + *b + *c;
}

// Test return multiples from class methods

class ClassA {
public:
  ClassA() {};
  int output2Input2Ret(int a, int b, int *c, int *d) {
    *c = a + 1;
    *d = b + 2;
    return *c + *d;
  }
  int inout2Input2Ret(int a, int b, int *c, int *d) {
    *c = *c + a;
    *d = *d + b;
    return *c + *d;
  }
};


%}
