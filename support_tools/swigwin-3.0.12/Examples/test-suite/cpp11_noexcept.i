%module cpp11_noexcept

%ignore NoExceptClass(NoExceptClass&&);
%rename(Assignment) NoExceptClass::operator=;

%inline %{

extern "C" void global_noexcept(int, bool) noexcept;

struct NoExceptClass {
  static const bool VeryTrue = true;

  NoExceptClass() noexcept {}
  NoExceptClass(const NoExceptClass&) noexcept {}
  NoExceptClass(NoExceptClass&&) noexcept {}
  NoExceptClass& operator=(const NoExceptClass&) noexcept { return *this; }
  ~NoExceptClass() noexcept {}

  void noex0() noexcept {}
  void noex1() noexcept(sizeof(int) == 4) {}
  void noex2() noexcept(true) {}
  void noex3() noexcept(false) {}
  void noex4() noexcept(VeryTrue) {}

  template<typename T> void template_noexcept(T) noexcept {}

  void noo1() const noexcept {}
  static void noo2() noexcept {}
  virtual void noo3() const noexcept {}
  virtual void noo4() const noexcept = delete;
  virtual void noo5() const throw() = delete;
};

struct NoExceptAbstract {
  virtual void noo4() const noexcept = 0;
  virtual ~NoExceptAbstract() noexcept = 0;
};

struct NoExceptDefaultDelete {
  template<typename T> NoExceptDefaultDelete(T) noexcept = delete;
  NoExceptDefaultDelete() noexcept = default;
  NoExceptDefaultDelete(const NoExceptDefaultDelete&) noexcept = delete;
  NoExceptDefaultDelete(NoExceptDefaultDelete&&) = delete;
  NoExceptDefaultDelete& operator=(const NoExceptDefaultDelete&) = delete;
  ~NoExceptDefaultDelete() noexcept = default;
};

%}

