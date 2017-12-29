package main

import . "./overload_template_fast"

func main() {
	_ = Foo()

	_ = Maximum(3, 4)
	_ = Maximum(3.4, 5.2)

	// mix 1
	if Mix1("hi") != 101 {
		panic("mix1(const char*)")
	}

	if Mix1(1.0, 1.0) != 102 {
		panic("mix1(double, const double &)")
	}

	if Mix1(1.0) != 103 {
		panic("mix1(double)")
	}

	// mix 2
	if Mix2("hi") != 101 {
		panic("mix2(const char*)")
	}

	if Mix2(1.0, 1.0) != 102 {
		panic("mix2(double, const double &)")
	}

	if Mix2(1.0) != 103 {
		panic("mix2(double)")
	}

	// mix 3
	if Mix3("hi") != 101 {
		panic("mix3(const char*)")
	}

	if Mix3(1.0, 1.0) != 102 {
		panic("mix3(double, const double &)")
	}

	if Mix3(1.0) != 103 {
		panic("mix3(double)")
	}

	// Combination 1
	if Overtparams1(100) != 10 {
		panic("overtparams1(int)")
	}

	if Overtparams1(100.0, 100) != 20 {
		panic("overtparams1(double, int)")
	}

	// Combination 2
	if Overtparams2(100.0, 100) != 40 {
		panic("overtparams2(double, int)")
	}

	// Combination 3
	if Overloaded() != 60 {
		panic("overloaded()")
	}

	if Overloaded(100.0, 100) != 70 {
		panic("overloaded(double, int)")
	}

	// Combination 4
	if Overloadedagain("hello") != 80 {
		panic("overloadedagain(const char *)")
	}

	if Overloadedagain() != 90 {
		panic("overloadedagain(double)")
	}

	// specializations
	if Specialization(10) != 202 {
		panic("specialization(int)")
	}

	if Specialization(10.0) != 203 {
		panic("specialization(double)")
	}

	if Specialization(10, 10) != 204 {
		panic("specialization(int, int)")
	}

	if Specialization(10.0, 10.0) != 205 {
		panic("specialization(double, double)")
	}

	if Specialization("hi", "hi") != 201 {
		panic("specialization(const char *, const char *)")
	}

	// simple specialization
	Xyz()
	Xyz_int()
	Xyz_double()

	// a bit of everything
	if Overload("hi") != 0 {
		panic("overload()")
	}

	if Overload(1) != 10 {
		panic("overload(int t)")
	}

	if Overload(1, 1) != 20 {
		panic("overload(int t, const int &)")
	}

	if Overload(1, "hello") != 30 {
		panic("overload(int t, const char *)")
	}

	k := NewKlass()
	if Overload(k) != 10 {
		panic("overload(Klass t)")
	}

	if Overload(k, k) != 20 {
		panic("overload(Klass t, const Klass &)")
	}

	if Overload(k, "hello") != 30 {
		panic("overload(Klass t, const char *)")
	}

	if Overload(10.0, "hi") != 40 {
		panic("overload(double t, const char *)")
	}

	if Overload() != 50 {
		panic("overload(const char *)")
	}

	// everything put in a namespace
	if Nsoverload("hi") != 1000 {
		panic("nsoverload()")
	}

	if Nsoverload(1) != 1010 {
		panic("nsoverload(int t)")
	}

	if Nsoverload(1, 1) != 1020 {
		panic("nsoverload(int t, const int &)")
	}

	if Nsoverload(1, "hello") != 1030 {
		panic("nsoverload(int t, const char *)")
	}

	if Nsoverload(k) != 1010 {
		panic("nsoverload(Klass t)")
	}

	if Nsoverload(k, k) != 1020 {
		panic("nsoverload(Klass t, const Klass &)")
	}

	if Nsoverload(k, "hello") != 1030 {
		panic("nsoverload(Klass t, const char *)")
	}

	if Nsoverload(10.0, "hi") != 1040 {
		panic("nsoverload(double t, const char *)")
	}

	if Nsoverload() != 1050 {
		panic("nsoverload(const char *)")
	}

	AFoo(1)
	b := NewB()
	b.Foo(1)
}
