
# This is the union runtime testcase. It ensures that values within a 
# union embedded within a struct can be set and read correctly.

if [ catch { load ./unions[info sharedlibextension] unions} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

# Create new instances of SmallStruct and BigStruct for later use
SmallStruct small
small configure -jill 200

BigStruct big
big configure -smallstruct [small cget -this]
big configure -jack 300

# Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
# Ensure values in EmbeddedUnionTest are set correctly for each.
EmbeddedUnionTest eut

# First check the SmallStruct in EmbeddedUnionTest
eut configure -number 1

#eut.uni.small = small
EmbeddedUnionTest_uni_small_set [EmbeddedUnionTest_uni_get [eut cget -this] ] [small cget -this]

#Jill1 = eut.uni.small.jill
set Jill1 [SmallStruct_jill_get [EmbeddedUnionTest_uni_small_get [EmbeddedUnionTest_uni_get [eut cget -this] ] ] ]
if {$Jill1 != 200} {
    puts stderr "Runtime test1 failed. eut.uni.small.jill=$Jill1"
    exit 1
}

set Num1 [eut cget -number]
if {$Num1 != 1} {
    puts stderr "Runtime test2 failed. eut.number=$Num1"
    exit 1
}

# Secondly check the BigStruct in EmbeddedUnionTest
eut configure -number 2
#eut.uni.big = big
EmbeddedUnionTest_uni_big_set [EmbeddedUnionTest_uni_get [eut cget -this] ] [big cget -this]
#Jack1 = eut.uni.big.jack
set Jack1 [BigStruct_jack_get [EmbeddedUnionTest_uni_big_get [EmbeddedUnionTest_uni_get [eut cget -this] ] ] ]
if {$Jack1 != 300} {
    puts stderr "Runtime test3 failed. eut.uni.big.jack=$Jack1"
    exit 1
}

#Jill2 = eut.uni.big.smallstruct.jill
set Jill2 [SmallStruct_jill_get [BigStruct_smallstruct_get [EmbeddedUnionTest_uni_big_get [EmbeddedUnionTest_uni_get [eut cget -this] ] ] ] ]
if {$Jill2 != 200} {
    puts stderr "Runtime test4 failed. eut.uni.big.smallstruct.jill=$Jill2"
    exit 1
}

set Num2 [eut cget -number]
if {$Num2 != 2} {
    puts stderr "Runtime test5 failed. eut.number=$Num2"
    exit 1
}

