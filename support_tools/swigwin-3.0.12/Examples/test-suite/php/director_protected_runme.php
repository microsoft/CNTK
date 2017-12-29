<?php

require "tests.php";
require "director_protected.php";

check::functions(array(foo_pong,foo_s,foo_q,foo_ping,foo_pang,foo_used,foo_cheer,bar_create,bar_callping,bar_callcheer,bar_cheer,bar_pong,bar_used,bar_ping,bar_pang,a_draw,b_draw));
check::classes(array(Foo,Bar,PrivateFoo,A,B,AA,BB));
check::globals(array(bar_a));

class FooBar extends Bar {
  protected function ping() {
    return "FooBar::ping();";
  }
}

class FooBar2 extends Bar {
  function ping() {
    return "FooBar2::ping();";
  }

  function pang() {
    return "FooBar2::pang();";
  }
}

class FooBar3 extends Bar {
  function cheer() {
    return "FooBar3::cheer();";
  }
}

$b = new Bar();
$f = $b->create();
$fb = new FooBar();
$fb2 = new FooBar2();
$fb3 = new FooBar3();

check::equal($fb->used(), "Foo::pang();Bar::pong();Foo::pong();FooBar::ping();", "bad FooBar::used");

check::equal($fb2->used(), "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();", "bad FooBar2::used");

check::equal($b->pong(), "Bar::pong();Foo::pong();Bar::ping();", "bad Bar::pong");

check::equal($f->pong(), "Bar::pong();Foo::pong();Bar::ping();", "bad Foo::pong");

check::equal($fb->pong(), "Bar::pong();Foo::pong();FooBar::ping();", "bad FooBar::pong");

$method = new ReflectionMethod('Bar', 'ping');
check::equal($method->isProtected(), true, "Foo::ping should be protected");

$method = new ReflectionMethod('Foo', 'ping');
check::equal($method->isProtected(), true, "Foo::ping should be protected");

$method = new ReflectionMethod('FooBar', 'pang');
check::equal($method->isProtected(), true, "FooBar::pang should be protected");

$method = new ReflectionMethod('Bar', 'cheer');
check::equal($method->isProtected(), true, "Bar::cheer should be protected");

$method = new ReflectionMethod('Foo', 'cheer');
check::equal($method->isProtected(), true, "Foo::cheer should be protected");

check::equal($fb3->cheer(), "FooBar3::cheer();", "bad fb3::pong");
check::equal($fb2->callping(), "FooBar2::ping();", "bad fb2::callping");
check::equal($fb2->callcheer(), "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();", "bad fb2::callcheer");
check::equal($fb3->callping(), "Bar::ping();", "bad fb3::callping");
check::equal($fb3->callcheer(), "FooBar3::cheer();", "bad fb3::callcheer");

check::done();
?>
