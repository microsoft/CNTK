/* This interface file checks whether the SWIG parser handles static
   array members of classes.  Bug reported by Annalisa Terracina
   <annalisa.terracina@datamat.it> on 2001-07-03. 
*/

%module static_array_member

%inline %{
class RB {
  static char *rberror[];
};
%}
