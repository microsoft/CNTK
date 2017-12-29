%module types_directive

#if defined(SWIGR)
// Avoid conflict with Date class in R
#define Date DateSwig
%inline %{
#define Date DateSwig
%}
#endif

%ignore Time2::operator Date *;

%inline %{
struct Date {
  Date(unsigned int year, unsigned int month, unsigned int day) : year(year), month(month), day(day) {}
  unsigned int year;
  unsigned int month;
  unsigned int day;
};

struct Time1 {
  Time1(unsigned int year, unsigned int month, unsigned int day, unsigned int seconds) : date(year, month, day), seconds(seconds) {}
  Date &dateFromTime() {
    return date;
  }
  Date date;
  unsigned int seconds;
};

struct Time2 {
  Time2(unsigned int year, unsigned int month, unsigned int day, unsigned int seconds) : date(year, month, day), seconds(seconds) {}
  operator Date *() {
    return &date;
  }
  Date date;
  unsigned int seconds;
};
Date add(const Date &date, unsigned int days) {
  Date newDate = date;
  newDate.day += days;
  return newDate;
}
%}

// allow conversion from Date -> Time1 using the following code
%types(Time1 = Date) %{
  Time1 *t = (Time1 *)$from;
  Date &d = t->dateFromTime();
  return (void *) &d;
%}

// allow conversion from Date -> Time2 using conversion operator (cast) in Time2
%types(Time2 = Date);

