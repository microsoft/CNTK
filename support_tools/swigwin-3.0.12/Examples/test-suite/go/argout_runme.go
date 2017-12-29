package main

import wrap "./argout"

func main() {
	ip := wrap.New_intp()
	wrap.Intp_assign(ip, 42)
	if r := wrap.Incp(ip); r != 42 {
		panic(r)
	}
	if r := wrap.Intp_value(ip); r != 43 {
		panic(r)
	}

	p := wrap.New_intp()
	wrap.Intp_assign(p, 2)
	if r := wrap.Incp(p); r != 2 {
		panic(r)
	}
	if r := wrap.Intp_value(p); r != 3 {
		panic(r)
	}

	r := wrap.New_intp()
	wrap.Intp_assign(r, 7)
	if r := wrap.Incr(r); r != 7 {
		panic(r)
	}
	if r := wrap.Intp_value(r); r != 8 {
		panic(r)
	}

	tr := wrap.New_intp()
	wrap.Intp_assign(tr, 4)
	if r := wrap.Inctr(tr); r != 4 {
		panic(r)
	}
	if r := wrap.Intp_value(tr); r != 5 {
		panic(r)
	}
}
