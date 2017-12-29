%module rename_strip_encoder

// strip the wx prefix from all identifiers
%rename("%(strip:[wx])s") ""; 

%inline %{

class wxSomeWidget {
};

struct wxAnotherWidget {
    void wxDoSomething() {}
};


%}
