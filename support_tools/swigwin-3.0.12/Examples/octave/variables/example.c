/* File : example.c */

/* I'm a file containing some C global variables */

/* Deal with Microsoft's attempt at deprecating C standard runtime functions */
#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER)
# define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdio.h>
#include <stdlib.h>
#include "example.h"

int              ivar = 0;
short            svar = 0;
long             lvar = 0;
unsigned int     uivar = 0;
unsigned short   usvar = 0;
unsigned long    ulvar = 0;
signed char      scvar = 0;
unsigned char    ucvar = 0;
char             cvar = 0;
float            fvar = 0;
double           dvar = 0;
char            *strvar = 0;
const char       cstrvar[] = "Goodbye";
int             *iptrvar = 0;
char             name[256] = "Dave";
char             path[256] = "/home/beazley";


/* Global variables involving a structure */
Point           *ptptr = 0;
Point            pt = { 10, 20 };

/* A variable that we will make read-only in the interface */
int              status = 1;

/* A debugging function to print out their values */

void print_vars() {
  printf("ivar      = %d\n", ivar);
  printf("svar      = %d\n", svar);
  printf("lvar      = %ld\n", lvar);
  printf("uivar     = %u\n", uivar);
  printf("usvar     = %u\n", usvar);
  printf("ulvar     = %lu\n", ulvar);
  printf("scvar     = %d\n", scvar);
  printf("ucvar     = %u\n", ucvar);
  printf("fvar      = %g\n", fvar);
  printf("dvar      = %g\n", dvar);
  printf("cvar      = %c\n", cvar);
  printf("strvar    = %s\n", strvar ? strvar : "(null)");
  printf("cstrvar   = %s\n", cstrvar);
  printf("iptrvar   = %p\n", (void *)iptrvar);
  printf("name      = %s\n", name);
  printf("ptptr     = %p (%d, %d)\n", (void *)ptptr, ptptr ? ptptr->x : 0, ptptr ? ptptr->y : 0);
  printf("pt        = (%d, %d)\n", pt.x, pt.y);
  printf("status    = %d\n", status);
}

/* A function to create an integer (to test iptrvar) */

int *new_int(int value) {
  int *ip = (int *) malloc(sizeof(int));
  *ip = value;
  return ip;
}

/* A function to create a point */

Point *new_Point(int x, int y) {
  Point *p = (Point *) malloc(sizeof(Point));
  p->x = x;
  p->y = y;
  return p;
}

char * Point_print(Point *p) {
  static char buffer[256];
  if (p) {
    sprintf(buffer,"(%d,%d)", p->x,p->y);
  } else {
    sprintf(buffer,"null");
  }
  return buffer;
}

void pt_print() {
  printf("(%d, %d)\n", pt.x, pt.y);
}
