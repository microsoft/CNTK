%module simple_array

extern int x[10];
extern double y[7];


%inline %{

struct BarArray {
  int i;
  double d;
};

extern struct BarArray bars[2]; 

int x[10];
double y[7];
struct BarArray bars[2]; 

void
initArray()
{
  int i, n;

  n = sizeof(x)/sizeof(x[0]);
  for(i = 0; i < n; i++) 
    x[i] = i;

  n = sizeof(y)/sizeof(y[0]);
  for(i = 0; i < n; i++) 
    y[i] = ((double) i)/ ((double) n);

  n = sizeof(bars)/sizeof(bars[0]);
  for(i = 0; i < n; i++)  {
    bars[i].i = x[i+2];
    bars[i].d = y[i+2];
  }

  return;
}

%}
