package main

import (
	"go_subdir_import_a"
	"testdir/go_subdir_import/go_subdir_import_b"
	"testdir/go_subdir_import/go_subdir_import_c"
)

func main() {
	b := go_subdir_import_b.NewObjB();
	c := go_subdir_import_c.NewObjC();
	v := go_subdir_import_a.AddFive(b, c)
	if v != 50 {
		panic(0)
	}
}
