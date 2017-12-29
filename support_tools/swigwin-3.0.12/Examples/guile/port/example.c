#include <stdio.h>
#include <errno.h>

void print_int(FILE *f, int i)
{
  if (fprintf(f, "%d\n", i)<0)
    perror("print_int");
}

int read_int(FILE *f)
{
  int i;
  if (fscanf(f, "%d", &i)!=1) {
    fprintf(stderr, "read_int: error reading from file\n");
    perror("read_int");
  }
  return i;
}
