var constover = require("constover");

p = constover.test("test");
if (p != "test") {
    throw "test failed!";
}

p = constover.test_pconst("test");
if (p != "test_pconst") {
    throw "test_pconst failed!";
}

f = new constover.Foo();

p = f.test("test");
if (p != "test") {
    throw "member-test failed!";
}

p = f.test_pconst("test");
if (p != "test_pconst") {
    throw "member-test_pconst failed!";
}

p = f.test_constm("test");
if (p != "test_constmethod") {
    throw "member-test_constm failed!";
}

p = f.test_pconstm("test");
if (p != "test_pconstmethod") {
    throw "member-test_pconstm failed!";
}
