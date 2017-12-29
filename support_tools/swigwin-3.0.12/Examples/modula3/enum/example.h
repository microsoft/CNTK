/* File : example.h */

#define PI 3.141

#define DAY_MONDAY    0
#define DAY_TUESDAY   1
#define DAY_WEDNESDAY 2
#define DAY_THURSDAY  3
#define DAY_FRIDAY    4
#define DAY_SATURDAY  5
#define DAY_SUNDAY    6

enum color { BLUE, RED, GREEN };

#define CLB_BLACK   0
#define CLB_BLUE    1
#define CLB_RED     2
#define CLB_MAGENTA 3
#define CLB_GREEN   4
#define CLB_CYAN    5
#define CLB_YELLOW  6
#define CLB_WHITE   7

/* Using this would be good style
   which cannot be expected for general C header files.
   Instead I want to demonstrate how to live without it.
enum month {
  MTHF_JANUARY,
  MTHF_FEBRUARY,
  MTHF_MARCH,
  MTHF_APRIL,
  MTHF_MAY,
  MTHF_JUNE,
  MTHF_JULY,
  MTHF_AUGUST,
  MTHF_SEPTEMBER,
  MTHF_OCTOBER,
  MTHF_NOVEMBER,
  MTHF_DECEMBER,
}
*/

/* Since there are no compile time constants in C / C++
   it is a common abuse
   to declare bit set (flag) constants
   as enumerations. */
enum calendar {
  MTHB_JANUARY   = 1 <<  0,     /* 1 << MTHF_JANUARY, */
  MTHB_FEBRUARY  = 1 <<  1,     /* 1 << MTHF_FEBRUARY, */
  MTHB_MARCH     = 1 <<  2,     /* 1 << MTHF_MARCH, */
  MTHB_APRIL     = 1 <<  3,     /* 1 << MTHF_APRIL, */
  MTHB_MAY       = 1 <<  4,     /* 1 << MTHF_MAY, */
  MTHB_JUNE      = 1 <<  5,     /* 1 << MTHF_JUNE, */
  MTHB_JULY      = 1 <<  6,     /* 1 << MTHF_JULY, */
  MTHB_AUGUST    = 1 <<  7,     /* 1 << MTHF_AUGUST, */
  MTHB_SEPTEMBER = 1 <<  8,     /* 1 << MTHF_SEPTEMBER, */
  MTHB_OCTOBER   = 1 <<  9,     /* 1 << MTHF_OCTOBER, */
  MTHB_NOVEMBER  = 1 << 10,     /* 1 << MTHF_NOVEMBER, */
  MTHB_DECEMBER  = 1 << 11,     /* 1 << MTHF_DECEMBER, */

  MTHB_SPRING    = MTHB_MARCH     | MTHB_APRIL   | MTHB_MAY,
  MTHB_SUMMER    = MTHB_JUNE      | MTHB_JULY    | MTHB_AUGUST,
  MTHB_AUTUMN    = MTHB_SEPTEMBER | MTHB_OCTOBER | MTHB_NOVEMBER,
  MTHB_WINTER    = MTHB_DECEMBER  | MTHB_JANUARY | MTHB_FEBRUARY,
};


namespace Answer {
  enum {
    UNIVERSE_AND_EVERYTHING = 42,
    SEVENTEEN_AND_FOUR = 21,
    TWOHUNDRED_PERCENT_OF_NOTHING = 0,
  };

  class Foo {
   public:
    Foo() { }
    enum speed { IMPULSE  = -2, WARP = 0, HYPER, LUDICROUS = 3};
    void enum_test(speed s);
  };
};

void enum_test(color c, Answer::Foo::speed s);
