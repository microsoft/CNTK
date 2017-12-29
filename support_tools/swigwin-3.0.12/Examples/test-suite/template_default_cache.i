%module template_default_cache;

%inline %{
    namespace d {
        template< typename T > class d {};
    }
%}

%ignore ns_a::iface1::Model;

%inline %{
namespace ns_a {
    namespace iface1 {
        class Model {};
        typedef d::d<Model> ModelPtr;
    }
    using iface1::ModelPtr;
}
%}

%inline %{
namespace ns_b {
    namespace iface1 {
        class Model {
        public:
            ns_a::ModelPtr foo() { return ns_a::ModelPtr(); };
        };
        typedef d::d<Model> ModelPtr;
        ns_a::ModelPtr get_mp_a() { return ns_a::ModelPtr(); }
        ModelPtr get_mp_b() { return ModelPtr(); }
    }
 }
%}
%template(AModelPtr) d::d<ns_a::iface1::Model>;
%template(BModelPtr) d::d<ns_b::iface1::Model>;
