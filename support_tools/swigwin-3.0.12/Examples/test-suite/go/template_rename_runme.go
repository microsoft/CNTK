package main

import "./template_rename"

func main() {
	i := template_rename.NewIFoo()
	d := template_rename.NewDFoo()

	_ = i.Blah_test(4)
	_ = i.Spam_test(5)
	_ = i.Groki_test(6)

	_ = d.Blah_test(7)
	_ = d.Spam(8)
	_ = d.Grok_test(9)
}
