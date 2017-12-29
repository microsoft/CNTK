var example = require("example");

console.log("Trying to catch some exceptions.");
t = new example.Test();
try{
  t.unknown();
  throw -1;
} catch(error)
{
  if(error == -1) {
    console.log("t.unknown() didn't throw");
  } else {
    console.log("successfully catched throw in Test::unknown().");
  }
}

try{
    t.simple();
    throw -1;
}
catch(error){
  if(error == -1) {
    console.log("t.simple() did not throw");
  } else {
    console.log("successfully catched throw in Test::simple().");
  }
}

try{
  t.message();
  throw -1;
} catch(error){
  if(error == -1) {
    console.log("t.message() did not throw");
  } else {
    console.log("successfully catched throw in Test::message().");
  }
}
    
try{
  t.hosed();
  throw -1;
}
catch(error){ 
  if(error == -1) {
    console.log("t.hosed() did not throw");
  } else {
    console.log("successfully catched throw in Test::hosed().");
  }
}

for (var i=1; i<4; i++) {
  try{
      t.multi(i);
      throw -1;
  }
  catch(error){
    if(error == -1) {
      console.log("t.multi(" + i + ") did not throw");
    } else {
      console.log("successfully catched throw in Test::multi().");
    }
  }
}    
