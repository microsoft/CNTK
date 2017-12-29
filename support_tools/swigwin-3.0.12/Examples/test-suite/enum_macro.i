%module enum_macro

%inline %{

#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* comma at end of enumerator list [-Werror=pedantic] */
#pragma GCC diagnostic ignored "-Wpedantic"
#endif


enum Greeks1
{
#define GREEK1 -1
    alpha1=1,
    beta1,
    theta1
};

enum Greeks2
{
    alpha2 = 2,
#define GREEK2 -2
    beta2,
    theta2
};

enum Greeks3
{
    alpha3,
    beta3,
#define GREEK3 -3
    theta3
};

enum Greeks4
{
    alpha4 = 4,
    beta4 = 5,
    theta4 = 6
#define GREEK4 -4
};

enum Greeks5
{
#define GREEK5 -5
    alpha5,
    beta5,
};

enum Greeks6
{
    alpha6,
#define GREEK6 -6
    beta6,
};

enum Greeks7
{
    alpha7,
    beta7,
#define GREEK7 -7
};

enum Greeks8
{
#define GREEK8 -8
    theta8
};

enum Greeks9
{
    theta9
#define GREEK9 -9
};

enum Greeks10
{
#define GREEK10 -10
    theta10,
};

enum Greeks11
{
    theta11,
#define GREEK11 -11
};

typedef enum {
    theta12 = 0
#define GREEK12 -12
} Greeks12;
%}


enum Greeks13
{
#define GREEK13 -13
};

