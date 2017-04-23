#include "logreg.h"

#include <string>
#include <iostream>
using namespace logreg;

int main(int argc, char* argv[]) {
  LogReg<float> lr(argv[1]);
    
  lr.Train();
  
  return 0;
}