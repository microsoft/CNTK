%module abstract_typedef


%inline %{
    struct Engine
    {
    };

    struct AbstractBaseClass
    {
      virtual ~AbstractBaseClass()
      {
      }
      
      virtual bool write(Engine& archive) const = 0;
    };    

    typedef Engine PersEngine;
    typedef AbstractBaseClass PersClassBase;      

    
    class A : public PersClassBase
    {
      // This works always
      // bool write(Engine& archive) const;

      // This doesn't with Swig 1.3.17.
      // But it works fine with 1.3.16
      bool write(PersEngine& archive) const
      {
	return true;
      }
      
    
    };
      
%}


/*

Problem related to the direct comparison of strings
in the file allocate.cxx (line 55)

          ......
	  String *local_decl = Getattr(dn,"decl");
	  if (local_decl && !Strcmp(local_decl, base_decl)) {
          ......

with the direct string comparison, no equivalent types
are checked and the two 'write' functions appear to be
different because

  "q(const).f(r.bss::PersEngine)." != "q(const).f(r.bss::Engine)."

*/
