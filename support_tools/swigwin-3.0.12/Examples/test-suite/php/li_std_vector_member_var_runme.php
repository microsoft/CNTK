<?php

require "tests.php";
require "li_std_vector_member_var.php";

$t = new Test();

check::equal($t->x, 0, "Test::x != 0");
check::equal($t->v->size(), 0, "Test::v.size() != 0");

$t->f(1);
check::equal($t->x, 1, "Test::x != 1");
check::equal($t->v->size(), 1, "Test::v.size() != 1");

$t->f(2);
check::equal($t->x, 3, "Test::x != 3");
check::equal($t->v->size(), 2, "Test::v.size() != 2");

$t->f(3);
check::equal($t->x, 6, "Test::x != 6");
check::equal($t->v->size(), 3, "Test::v.size() != 3");

$T = new T();
$T->start_t = new S();
$T->length = 7;
check::equal($T->start_t->x, 4, "S::x != 4");
check::equal($T->length, 7, "T::length != 7");

check::done();
?>
