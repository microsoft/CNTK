#include "common.h"

#include <string.h>
#include <stdio.h>

using namespace multiverso::test;

void PrintUsage() {
  printf("Usage: multiverso.test kv|array|net|matrix|allreduce\n");
}

int main(int argc, char* argv[]) {
  if (argc != 2) PrintUsage();
  else {
    if (strcmp(argv[1], "kv") == 0) TestKV(argc, argv);
    else if (strcmp(argv[1], "array") == 0) TestArray(argc, argv);
    else if (strcmp(argv[1], "net") == 0) TestNet(argc, argv);
    else if (strcmp(argv[1], "matrix") == 0) TestMatrix(argc, argv);
    else if (strcmp(argv[1], "allreduce") == 0) TestAllreduce(argc, argv);
    else {
      PrintUsage();
    }
  }
  return 0;
}
