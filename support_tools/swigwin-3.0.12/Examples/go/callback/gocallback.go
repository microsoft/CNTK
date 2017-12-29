package example

import (
	"fmt"
)

type GoCallback interface {
	Callback
	deleteCallback()
	IsGoCallback()
}

type goCallback struct {
	Callback
}

func (p *goCallback) deleteCallback() {
	DeleteDirectorCallback(p.Callback)
}

func (p *goCallback) IsGoCallback() {}

type overwrittenMethodsOnCallback struct {
	p Callback
}

func NewGoCallback() GoCallback {
	om := &overwrittenMethodsOnCallback{}
	p := NewDirectorCallback(om)
	om.p = p

	return &goCallback{Callback: p}
}

func DeleteGoCallback(p GoCallback) {
	p.deleteCallback()
}

func (p *goCallback) Run() {
	fmt.Println("GoCallback.Run")
}
