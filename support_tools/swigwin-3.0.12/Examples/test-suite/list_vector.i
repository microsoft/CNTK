/* -*- c -*- */

%module list_vector

%include "list-vector.i"

%multiple_values

/* The ordinary, well behaved multi-typemap. */
double sum_list(int LISTLENINPUT, double *LISTINPUT);
double sum_vector(int VECTORLENINPUT, double *VECTORINPUT);
void one_to_seven_list(int *LISTLENOUTPUT, int **LISTOUTPUT);
void one_to_seven_vector(int *VECTORLENOUTPUT, int **VECTOROUTPUT);

/* Variants with `size_t' instead of `int' length.  */
double sum_list2(size_t LISTLENINPUT, double *LISTINPUT);
double sum_vector2(size_t VECTORLENINPUT, double *VECTORINPUT);
void one_to_seven_list2(size_t *LISTLENOUTPUT, int **LISTOUTPUT);
void one_to_seven_vector2(size_t *VECTORLENOUTPUT, int **VECTOROUTPUT);

/* Parallel variants */

double sum_lists(int PARALLEL_LISTLENINPUT,
		 double *PARALLEL_LISTINPUT,
		 int *PARALLEL_LISTINPUT,
		 int *PARALLEL_LISTINPUT);
double sum_lists2(size_t PARALLEL_LISTLENINPUT,
		 double *PARALLEL_LISTINPUT,
		 int *PARALLEL_LISTINPUT,
		 int *PARALLEL_LISTINPUT);
void produce_lists(int *PARALLEL_VECTORLENOUTPUT,
		   int **PARALLEL_VECTOROUTPUT,
		   int **PARALLEL_VECTOROUTPUT,
		   double **PARALLEL_VECTOROUTPUT);

%{
  double sum_list(int length, double *item)
  {
    int i;
    double res = 0.0;
    for (i = 0; i<length; i++)
      res += item[i];
    return res;
  }

  double sum_list2(size_t length, double *item)
  {
    size_t i;
    double res = 0.0;
    for (i = 0; i<length; i++)
      res += item[i];
    return res;
  }


  double sum_vector(int length, double *item)
  {
    int i;
    double res = 0.0;
    for (i = 0; i<length; i++)
      res += item[i];
    return res;
  }

  double sum_vector2(size_t length, double *item)
  {
    size_t i;
    double res = 0.0;
    for (i = 0; i<length; i++)
      res += item[i];
    return res;
  }


  void one_to_seven_list(int *length_p, int **list_p)
  {
    int i;
    *length_p = 7;
    *list_p = malloc(7 * sizeof(int));
    for (i = 0; i<7; i++)
      (*list_p)[i] = i+1;
  }

   void one_to_seven_list2(size_t *length_p, int **list_p)
  {
    size_t i;
    *length_p = 7;
    *list_p = malloc(7 * sizeof(int));
    for (i = 0; i<7; i++)
      (*list_p)[i] = i+1;
  }

   void one_to_seven_vector(int *length_p, int **list_p)
  {
    int i;
    *length_p = 7;
    *list_p = malloc(7 * sizeof(int));
    for (i = 0; i<7; i++)
      (*list_p)[i] = i+1;
  }

  void one_to_seven_vector2(size_t *length_p, int **list_p)
  {
    size_t i;
    *length_p = 7;
    *list_p = malloc(7 * sizeof(int));
    for (i = 0; i<7; i++)
      (*list_p)[i] = i+1;
  }

double sum_lists(int len,
		 double *list1,
		 int *list2,
		 int *list3)
{
  int i;
  double sum = 0.0;
  for (i = 0; i<len; i++)
    sum += (list1[i] + list2[i] + list3[i]);
  return sum;
}

double sum_lists2(size_t len,
		  double *list1,
		  int *list2,
		  int *list3)
{
  size_t i;
  double sum = 0.0;
  for (i = 0; i<len; i++)
    sum += (list1[i] + list2[i] + list3[i]);
  return sum;
}

void produce_lists(int *len_p,
		   int **list1_p,
		   int **list2_p,
		   double **list3_p)
{
  int i;
  *len_p = 5;
  *list1_p = malloc(sizeof(int) * 5);
  *list2_p = malloc(sizeof(int) * 5);
  *list3_p = malloc(sizeof(double) * 5);
  for (i = 0; i<5; i++) {
    (*list1_p)[i] = i;
    (*list2_p)[i] = i*i;
    (*list3_p)[i] = 1.5*i;
  }
}

  
%}
