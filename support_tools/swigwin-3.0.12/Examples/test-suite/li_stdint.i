%module li_stdint

%include <stdint.i>

%inline %{
  struct StdI {
    int8_t   int8_member;
    int16_t  int16_member;
    int32_t  int32_member;
    int64_t  int64_member;
    uint8_t  uint8_member;
    uint16_t uint16_member;
    uint32_t uint32_member;
    uint64_t uint64_member;
  };

  int8_t   int8_td (int8_t  i) { return i; }
  int16_t  int16_td(int16_t i) { return i; }
  int32_t  int32_td(int32_t i) { return i; }
  int64_t  int64_td(int64_t i) { return i; }
  uint8_t  uint8_td (int8_t  i) { return i; }
  uint16_t uint16_td(int16_t i) { return i; }
  uint32_t uint32_td(int32_t i) { return i; }
  uint64_t uint64_td(int64_t i) { return i; }

  struct StdIf {
    int_fast8_t   int_fast8_member;
    int_fast16_t  int_fast16_member;
    int_fast32_t  int_fast32_member;
    int_fast64_t  int_fast64_member;
    uint_fast8_t  uint_fast8_member;
    uint_fast16_t uint_fast16_member;
    uint_fast32_t uint_fast32_member;
    uint_fast64_t uint_fast64_member;
  };

  int_fast8_t   int_fast8_td (int_fast8_t  i) { return i; }
  int_fast16_t  int_fast16_td(int_fast16_t i) { return i; }
  int_fast32_t  int_fast32_td(int_fast32_t i) { return i; }
  int_fast64_t  int_fast64_td(int_fast64_t i) { return i; }
  uint_fast8_t  uint_fast8_td (int_fast8_t  i) { return i; }
  uint_fast16_t uint_fast16_td(int_fast16_t i) { return i; }
  uint_fast32_t uint_fast32_td(int_fast32_t i) { return i; }
  uint_fast64_t uint_fast64_td(int_fast64_t i) { return i; }

  struct StdIl {
    int_least8_t   int_least8_member;
    int_least16_t  int_least16_member;
    int_least32_t  int_least32_member;
    int_least64_t  int_least64_member;
    uint_least8_t  uint_least8_member;
    uint_least16_t uint_least16_member;
    uint_least32_t uint_least32_member;
    uint_least64_t uint_least64_member;
  };

  int_least8_t   int_least8_td (int_least8_t  i) { return i; }
  int_least16_t  int_least16_td(int_least16_t i) { return i; }
  int_least32_t  int_least32_td(int_least32_t i) { return i; }
  int_least64_t  int_least64_td(int_least64_t i) { return i; }
  uint_least8_t  uint_least8_td (int_least8_t  i) { return i; }
  uint_least16_t uint_least16_td(int_least16_t i) { return i; }
  uint_least32_t uint_least32_td(int_least32_t i) { return i; }
  uint_least64_t uint_least64_td(int_least64_t i) { return i; }

%}

