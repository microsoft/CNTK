
if [ catch { load ./cpp11_strongly_typed_enumerations[info sharedlibextension] cpp11_strongly_typed_enumerations} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

proc enumCheck {actual expected} {
  if {$actual != $expected} {
    error "Enum value mismatch. Expected $expected. Actual: $actual"
  }
  return [expr $expected + 1]
}

#set val 0
#set val [enumCheck $Enum1_Val1  $val]
#set val [enumCheck $Enum1_Val2  $val]

set val 0
set val [enumCheck $Enum1_Val1 $val]
set val [enumCheck $Enum1_Val2 $val]
set val [enumCheck $Enum1_Val3 13]
set val [enumCheck $Enum1_Val4 $val]
set val [enumCheck $Enum1_Val5a 13]
set val [enumCheck $Enum1_Val6a $val]

set val 0
set val [enumCheck $Enum2_Val1 $val]
set val [enumCheck $Enum2_Val2 $val]
set val [enumCheck $Enum2_Val3 23]
set val [enumCheck $Enum2_Val4 $val]
set val [enumCheck $Enum2_Val5b 23]
set val [enumCheck $Enum2_Val6b $val]

set val 0
set val [enumCheck $Val1 $val]
set val [enumCheck $Val2 $val]
set val [enumCheck $Val3 43]
set val [enumCheck $Val4 $val]

set val 0
set val [enumCheck $Enum5_Val1 $val]
set val [enumCheck $Enum5_Val2 $val]
set val [enumCheck $Enum5_Val3 53]
set val [enumCheck $Enum5_Val4 $val]

set val 0
set val [enumCheck $Enum6_Val1 $val]
set val [enumCheck $Enum6_Val2 $val]
set val [enumCheck $Enum6_Val3 63]
set val [enumCheck $Enum6_Val4 $val]

set val 0
set val [enumCheck $Enum7td_Val1 $val]
set val [enumCheck $Enum7td_Val2 $val]
set val [enumCheck $Enum7td_Val3 73]
set val [enumCheck $Enum7td_Val4 $val]

set val 0
set val [enumCheck $Enum8_Val1 $val]
set val [enumCheck $Enum8_Val2 $val]
set val [enumCheck $Enum8_Val3 83]
set val [enumCheck $Enum8_Val4 $val]

set val 0
set val [enumCheck $Enum10_Val1 $val]
set val [enumCheck $Enum10_Val2 $val]
set val [enumCheck $Enum10_Val3 103]
set val [enumCheck $Enum10_Val4 $val]

set val 0
set val [enumCheck $Class1_Enum12_Val1 1121]
set val [enumCheck $Class1_Enum12_Val2 1122]
set val [enumCheck $Class1_Enum12_Val3 $val]
set val [enumCheck $Class1_Enum12_Val4 $val]
set val [enumCheck $Class1_Enum12_Val5c 1121]
set val [enumCheck $Class1_Enum12_Val6c $val]

set val 0
set val [enumCheck $Class1_Val1 1131]
set val [enumCheck $Class1_Val2 1132]
set val [enumCheck $Class1_Val3 $val]
set val [enumCheck $Class1_Val4 $val]
set val [enumCheck $Class1_Val5d 1131]
set val [enumCheck $Class1_Val6d $val]

set val 0
set val [enumCheck $Class1_Enum14_Val1 1141]
set val [enumCheck $Class1_Enum14_Val2 1142]
set val [enumCheck $Class1_Enum14_Val3 $val]
set val [enumCheck $Class1_Enum14_Val4 $val]
set val [enumCheck $Class1_Enum14_Val5e 1141]
set val [enumCheck $Class1_Enum14_Val6e $val]

# Requires nested class support to work
#set val 0
#set val [enumCheck $Class1_Struct1_Enum12_Val1 3121]
#set val [enumCheck $Class1_Struct1_Enum12_Val2 3122]
#set val [enumCheck $Class1_Struct1_Enum12_Val3 $val]
#set val [enumCheck $Class1_Struct1_Enum12_Val4 $val]
#set val [enumCheck $Class1_Struct1_Enum12_Val5f 3121]
#set val [enumCheck $Class1_Struct1_Enum12_Val6f $val]
#
#set val 0
#set val [enumCheck $Class1_Struct1_Val1 3131]
#set val [enumCheck $Class1_Struct1_Val2 3132]
#set val [enumCheck $Class1_Struct1_Val3 $val]
#set val [enumCheck $Class1_Struct1_Val4 $val]
#
#set val 0
#set val [enumCheck $Class1_Struct1_Enum14_Val1 3141]
#set val [enumCheck $Class1_Struct1_Enum14_Val2 3142]
#set val [enumCheck $Class1_Struct1_Enum14_Val3 $val]
#set val [enumCheck $Class1_Struct1_Enum14_Val4 $val]
#set val [enumCheck $Class1_Struct1_Enum14_Val5g 3141]
#set val [enumCheck $Class1_Struct1_Enum14_Val6g $val]

set val 0
set val [enumCheck $Class2_Enum12_Val1 2121]
set val [enumCheck $Class2_Enum12_Val2 2122]
set val [enumCheck $Class2_Enum12_Val3 $val]
set val [enumCheck $Class2_Enum12_Val4 $val]
set val [enumCheck $Class2_Enum12_Val5h 2121]
set val [enumCheck $Class2_Enum12_Val6h $val]

set val 0
set val [enumCheck $Class2_Val1 2131]
set val [enumCheck $Class2_Val2 2132]
set val [enumCheck $Class2_Val3 $val]
set val [enumCheck $Class2_Val4 $val]
set val [enumCheck $Class2_Val5i 2131]
set val [enumCheck $Class2_Val6i $val]

set val 0
set val [enumCheck $Class2_Enum14_Val1 2141]
set val [enumCheck $Class2_Enum14_Val2 2142]
set val [enumCheck $Class2_Enum14_Val3 $val]
set val [enumCheck $Class2_Enum14_Val4 $val]
set val [enumCheck $Class2_Enum14_Val5j 2141]
set val [enumCheck $Class2_Enum14_Val6j $val]

# Requires nested class support to work
#set val 0
#set val [enumCheck $Class2_Struct1_Enum12_Val1 4121]
#set val [enumCheck $Class2_Struct1_Enum12_Val2 4122]
#set val [enumCheck $Class2_Struct1_Enum12_Val3 $val]
#set val [enumCheck $Class2_Struct1_Enum12_Val4 $val]
#set val [enumCheck $Class2_Struct1_Enum12_Val5k 4121]
#set val [enumCheck $Class2_Struct1_Enum12_Val6k $val]
#
#set val 0
#set val [enumCheck $Class2_Struct1_Val1 4131]
#set val [enumCheck $Class2_Struct1_Val2 4132]
#set val [enumCheck $Class2_Struct1_Val3 $val]
#set val [enumCheck $Class2_Struct1_Val4 $val]
#set val [enumCheck $Class2_Struct1_Val5l 4131]
#set val [enumCheck $Class2_Struct1_Val6l $val]
#
#set val 0
#set val [enumCheck $Class2_Struct1_Enum14_Val1 4141]
#set val [enumCheck $Class2_Struct1_Enum14_Val2 4142]
#set val [enumCheck $Class2_Struct1_Enum14_Val3 $val]
#set val [enumCheck $Class2_Struct1_Enum14_Val4 $val]
#set val [enumCheck $Class2_Struct1_Enum14_Val5m 4141]
#set val [enumCheck $Class2_Struct1_Enum14_Val6m $val]

set class1 [Class1]
enumCheck [$class1 class1Test1 $Enum1_Val5a] 13
enumCheck [$class1 class1Test2 $Class1_Enum12_Val5c] 1121
#enumCheck [$class1 class1Test3 $Class1_Struct1_Enum12_Val5f] 3121

enumCheck [globalTest1 $Enum1_Val5a] 13
enumCheck [globalTest2 $Class1_Enum12_Val5c] 1121
#enumCheck [globalTest3 $Class1_Struct1_Enum12_Val5f] 3121
