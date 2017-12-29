%module robin

%inline %{
const char *run(void) {
    return "AWAY!";
}
%}
