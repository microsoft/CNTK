%module rename_pcre_encoder

// strip the wx prefix from all identifiers except those starting with wxEVT
%rename("%(regex:/wx(?!EVT)(.*)/\\1/)s") "";

// Change "{Set,Get}Foo" naming convention to "{put,get}_foo" one.
%rename("%(regex:/^Set(.*)/put_\\l\\1/)s", %$isfunction) "";
%rename("%(regex:/^Get(.*)/get_\\l\\1/)s", %$isfunction) "";

// Make some words stand out (unfortunately we don't have "global" flag): we
// use \U to capitalize the second capture group and then \E to preserve the
// case of the rest.
%rename("%(regex:/(.*?)(nsa)(.*?)\\2(.*?)\\2(.*?)\\2(.*)/\\1\\U\\2\\E\\3\\U\\2\\E\\4\\U\\2\\E\\5\\U\\2\\E\\6/)s") "";

%inline %{

struct wxSomeWidget {
    void SetBorderWidth(int width) { m_width = width; }
    int GetBorderWidth() const { return m_width; }

    void SetSize(int, int) {}

    int m_width;
};

struct wxAnotherWidget {
    void DoSomething() {}
};

class wxEVTSomeEvent {
};

class xUnchangedName {
};

inline int StartInsaneAndUnsavoryTransatlanticRansack() { return 42; }

%}
