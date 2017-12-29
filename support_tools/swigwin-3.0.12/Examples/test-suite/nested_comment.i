%module nested_comment

#pragma SWIG nowarn=SWIGWARN_PARSE_UNNAMED_NESTED_CLASS

// this example shows a problem with 'dump_nested' (parser.y).

// bug #949654
%inline %{
  typedef struct s1 {
    union {
      int fsc; /* genie structure hiding - Conductor
                */
      int fso; /* genie structure hiding - FSOptions
                */
      struct {
        double *vals;
        int size;
      } vector_val; /* matrix values are mainly used
                       in rlgc models */
      char *name;
    } n ;
  } s2; 
%}

// comment in nested struct
%inline %{
struct a
{
  struct {
    /*struct*/
    struct {
      int b; /**< v1/v2 B-tree & local/fractal heap for groups, B-tree for chunked datasets */
    } c;
  } d;
};
%}
