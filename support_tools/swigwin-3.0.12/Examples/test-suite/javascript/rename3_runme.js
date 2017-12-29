var rename = require("rename3");

function part1() {
  var xyz = new rename.XYZInt();
  notxyz = new rename.NotXYZInt();
  xyz.opIntPtrA();
  xyz.opIntPtrB();
  xyz.opAnother2();
  xyz.opT2();
  xyz.tMethod2(0);
  xyz.tMethodNotXYZ2(notxyz);
  xyz.opNotXYZ2();
  xyz.opXYZ2();
}

function part2() {
  var xyz = new rename.XYZDouble();
  var notxyz = new rename.NotXYZDouble();
  xyz.opIntPtrA();
  xyz.opIntPtrB();
  xyz.opAnother1();
  xyz.opT1();
  xyz.tMethod1(0);
  xyz.tMethodNotXYZ1(notxyz);
  xyz.opNotXYZ1();
  xyz.opXYZ1();
}

function part3(){
  var xyz = new rename.XYZKlass();
  var notxyz = new rename.NotXYZKlass();
  xyz.opIntPtrA();
  xyz.opIntPtrB();
  xyz.opAnother3();
  xyz.opT3();
  xyz.tMethod3(new rename.Klass());
  xyz.tMethodNotXYZ3(notxyz);
  xyz.opNotXYZ3();
  xyz.opXYZ3();
}

function part4() {
  var xyz = new rename.XYZEnu();
  var notxyz = new rename.NotXYZEnu();
  xyz.opIntPtrA();
  xyz.opIntPtrB();
  xyz.opAnother4();
  xyz.opT4();
  xyz.tMethod4(rename.En1);
  xyz.tMethodNotXYZ4(notxyz);
  xyz.opNotXYZ4();
  xyz.opXYZ4();
}

function part5() {
  var abc = new rename.ABC();
  abc.methodABC(abc);
  var k = new rename.Klass();
  abc.methodKlass(k);
  var a = abc.opABC();
  k = abc.opKlass();
}

part1();
part2();
part3();
part4();
part5();
