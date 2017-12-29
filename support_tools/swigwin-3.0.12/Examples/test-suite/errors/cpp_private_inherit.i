%module xxx

class Foo {
};

class Bar : private Foo {
};

class Spam : protected Foo {
};

