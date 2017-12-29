package example

// FooBarGo is a superset of FooBarAbstract and hence FooBarGo can be used as a
// drop in replacement for FooBarAbstract but the reverse causes a compile time
// error.
type FooBarGo interface {
	FooBarAbstract
	deleteFooBarAbstract()
	IsFooBarGo()
}

// Via embedding fooBarGo "inherits" all methods of FooBarAbstract.
type fooBarGo struct {
	FooBarAbstract
}

func (fbgs *fooBarGo) deleteFooBarAbstract() {
	DeleteDirectorFooBarAbstract(fbgs.FooBarAbstract)
}

// The IsFooBarGo method ensures that FooBarGo is a superset of FooBarAbstract.
// This is also how the class hierarchy gets represented by the SWIG generated
// wrapper code.  For an instance FooBarCpp has the IsFooBarAbstract and
// IsFooBarCpp methods.
func (fbgs *fooBarGo) IsFooBarGo() {}

// Go type that defines the DirectorInterface. It contains the Foo and Bar
// methods that overwrite the respective virtual C++ methods on FooBarAbstract.
type overwrittenMethodsOnFooBarAbstract struct {
	// Backlink to FooBarAbstract so that the rest of the class can be used by
	// the overridden methods.
	fb FooBarAbstract

	// If additional constructor arguments have been given they are typically
	// stored here so that the overriden methods can use them.
}

func (om *overwrittenMethodsOnFooBarAbstract) Foo() string {
	// DirectorFooBarAbstractFoo calls the base method FooBarAbstract::Foo.
	return "Go " + DirectorFooBarAbstractFoo(om.fb)
}

func (om *overwrittenMethodsOnFooBarAbstract) Bar() string {
	return "Go Bar"
}

func NewFooBarGo() FooBarGo {
	// Instantiate FooBarAbstract with selected methods overridden.  The methods
	// that will be overwritten are defined on
	// overwrittenMethodsOnFooBarAbstract and have a compatible signature to the
	// respective virtual C++ methods. Furthermore additional constructor
	// arguments will be typically stored in the
	// overwrittenMethodsOnFooBarAbstract struct.
	om := &overwrittenMethodsOnFooBarAbstract{}
	fb := NewDirectorFooBarAbstract(om)
	om.fb = fb // Backlink causes cycle as fb.v = om!

	fbgs := &fooBarGo{FooBarAbstract: fb}
	// The memory of the FooBarAbstract director object instance can be
	// automatically freed once the FooBarGo instance is garbage collected by
	// uncommenting the following line.  Please make sure to understand the
	// runtime.SetFinalizer specific gotchas before doing this.  Furthemore
	// DeleteFooBarGo should be deleted if a finalizer is in use or the fooBarGo
	// struct needs additional data to prevent double deletion.
	// runtime.SetFinalizer(fbgs, FooBarGo.deleteFooBarAbstract)
	return fbgs
}

// Recommended to be removed if runtime.SetFinalizer is in use.
func DeleteFooBarGo(fbg FooBarGo) {
	fbg.deleteFooBarAbstract()
}
