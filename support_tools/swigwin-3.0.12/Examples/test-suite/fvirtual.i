// This testcase tests corner cases for the -fvirtual optimisation flag.
// Note that the test-suite does not actually run with -fvirtual at any point, but this can be tested using the SWIG_FEATURES=-fvirtual env variable.
%module fvirtual

// Test overloaded methods #1508327 (requires a scripting language runtime test)
%inline %{
  class Node {
    public:
      virtual int addChild( Node *child ) { return 1; }
      virtual ~Node() {}
  };

  class NodeSwitch : public Node {
    public :
      virtual int addChild( Node *child ) { return 2; } // This was hidden with -fvirtual
      virtual int addChild( Node *child, bool value ) { return 3; }
      virtual ~NodeSwitch() {}
  };
%}


