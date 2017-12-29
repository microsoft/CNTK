#include <stdio.h>

int Circle (int x, int y, int radius) {
  /* Draw Circle */
  printf("Drawing the circle...\n");
  /* Return -1 to test contract post assertion */
  if (radius == 2)
    return -1;
  else
    return 1;
}
