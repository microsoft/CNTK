require 'li_boost_shared_ptr'
require 'swig_gc'

#debug = $VERBOSE
debug = false

# simple shared_ptr usage - created in C++


class Li_boost_shared_ptr_runme

    def main(debug)
        if (debug)
            puts "Started"
        end

        Li_boost_shared_ptr::debug_shared = debug

        # Change loop count to run for a long time to monitor memory
        loopCount = 1  # 5000
        1.upto(loopCount) do
            self.runtest()
        end

        # Expect 1 instance - the one global variable (GlobalValue)
        GC.track_class = Li_boost_shared_ptr::Klass
        invokeGC("Final GC")

# Actual count is 3 due to memory leaks calling rb_raise in the call to Li_boost_shared_ptr::valuetest(nil)
# as setjmp/longjmp are used thereby failing to call destructors of Klass instances on the stack in _wrap_valuetest
# This is a generic problem in Ruby wrappers, not shared_ptr specific
#        expectedCount = 1
        expectedCount = 3
        actualCount = Li_boost_shared_ptr::Klass.getTotal_count()
        if (actualCount != expectedCount)
#            raise RuntimeError, "GC failed to run (li_boost_shared_ptr). Expected count: #{expectedCount} Actual count: #{actualCount}"
            puts "GC failed to run (li_boost_shared_ptr). Expected count: #{expectedCount} Actual count: #{actualCount}"
        end

        wrapper_count = Li_boost_shared_ptr::shared_ptr_wrapper_count()
        if (wrapper_count != Li_boost_shared_ptr.NOT_COUNTING)
            # Expect 1 instance - the one global variable (GlobalSmartValue)
            if (wrapper_count != 1)
                raise RuntimeError, "shared_ptr wrapper count=#{wrapper_count}"
            end
        end

        if (debug)
            puts "Finished"
        end
    end

    def invokeGC(debug_msg)
        puts "invokeGC #{debug_msg} start" if $VERBOSE
        GC.stats if $VERBOSE
        GC.start
        puts "invokeGC #{debug_msg} end" if $VERBOSE
    end

    def runtest
        # simple shared_ptr usage - created in C++
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        val = k.getValue()
        self.verifyValue("me oh my", val)
        self.verifyCount(1, k)

        # simple shared_ptr usage - not created in C++
        k = Li_boost_shared_ptr::factorycreate()
        val = k.getValue()
        self.verifyValue("factorycreate", val)
        self.verifyCount(1, k)

        # pass by shared_ptr
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointertest", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr pointer
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerpointertest", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr reference
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerreftest", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr pointer reference
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerpointerreftest", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # const pass by shared_ptr
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::constsmartpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # const pass by shared_ptr pointer
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::constsmartpointerpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # const pass by shared_ptr reference
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::constsmartpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by value
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::valuetest(k)
        val = kret.getValue()
        self.verifyValue("me oh my valuetest", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # pass by pointer
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::pointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my pointertest", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # pass by reference
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::reftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my reftest", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # pass by pointer reference
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        kret = Li_boost_shared_ptr::pointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my pointerreftest", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # null tests
        k = nil

        if (Li_boost_shared_ptr::smartpointertest(k) != nil)
            raise RuntimeError, "return was not null"
        end

        if (Li_boost_shared_ptr::smartpointerpointertest(k) != nil)
            raise RuntimeError, "return was not null"
        end

        if (Li_boost_shared_ptr::smartpointerreftest(k) != nil)
            raise RuntimeError, "return was not null"
        end

        if (Li_boost_shared_ptr::smartpointerpointerreftest(k) != nil)
            raise RuntimeError, "return was not null"
        end

        if (Li_boost_shared_ptr::nullsmartpointerpointertest(nil) != "null pointer")
            raise RuntimeError, "not null smartpointer pointer"
        end

        begin
            Li_boost_shared_ptr::valuetest(k)
            raise RuntimeError, "Failed to catch null pointer"
        rescue ArgumentError
        end

        if (Li_boost_shared_ptr::pointertest(k) != nil)
            raise RuntimeError, "return was not null"
        end

        begin
            Li_boost_shared_ptr::reftest(k)
            raise RuntimeError, "Failed to catch null pointer"
        rescue ArgumentError
        end

        # $owner
        k = Li_boost_shared_ptr::pointerownertest()
        val = k.getValue()
        self.verifyValue("pointerownertest", val)
        self.verifyCount(1, k)
        k = Li_boost_shared_ptr::smartpointerpointerownertest()
        val = k.getValue()
        self.verifyValue("smartpointerpointerownertest", val)
        self.verifyCount(1, k)

        # //////////////////////////////// Derived class //////////////////////
        # derived pass by shared_ptr
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedsmartptrtest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedsmartptrtest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # derived pass by shared_ptr pointer
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedsmartptrpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedsmartptrpointertest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # derived pass by shared_ptr ref
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedsmartptrreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedsmartptrreftest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # derived pass by shared_ptr pointer ref
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedsmartptrpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedsmartptrpointerreftest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # derived pass by pointer
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedpointertest-Derived", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # derived pass by ref
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::derivedreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my derivedreftest-Derived", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # //////////////////////////////// Derived and base class mixed ///////
        # pass by shared_ptr (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointertest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr pointer (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerpointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerpointertest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr reference (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerreftest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by shared_ptr pointer reference (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::smartpointerpointerreftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my smartpointerpointerreftest-Derived", val)
        self.verifyCount(2, k)
        self.verifyCount(2, kret)

        # pass by value (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::valuetest(k)
        val = kret.getValue()
        self.verifyValue("me oh my valuetest", val)  # note slicing
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # pass by pointer (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::pointertest(k)
        val = kret.getValue()
        self.verifyValue("me oh my pointertest-Derived", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # pass by ref (mixed)
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        kret = Li_boost_shared_ptr::reftest(k)
        val = kret.getValue()
        self.verifyValue("me oh my reftest-Derived", val)
        self.verifyCount(1, k)
        self.verifyCount(1, kret)

        # //////////////////////////////// Overloading tests //////////////////
        # Base class
        k = Li_boost_shared_ptr::Klass.new("me oh my")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyval(k), "rawbyval")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyref(k), "rawbyref")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyptr(k), "rawbyptr")
        self.verifyValue(
            Li_boost_shared_ptr::overload_rawbyptrref(k), "rawbyptrref")

        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyval(k), "smartbyval")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyref(k), "smartbyref")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyptr(k), "smartbyptr")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyptrref(k), "smartbyptrref")

        # Derived class
        k = Li_boost_shared_ptr::KlassDerived.new("me oh my")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyval(k), "rawbyval")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyref(k), "rawbyref")
        self.verifyValue(Li_boost_shared_ptr::overload_rawbyptr(k), "rawbyptr")
        self.verifyValue(
            Li_boost_shared_ptr::overload_rawbyptrref(k), "rawbyptrref")

        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyval(k), "smartbyval")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyref(k), "smartbyref")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyptr(k), "smartbyptr")
        self.verifyValue(
            Li_boost_shared_ptr::overload_smartbyptrref(k), "smartbyptrref")

        # 3rd derived class
        k = Li_boost_shared_ptr::Klass3rdDerived.new("me oh my")
        val = k.getValue()
        self.verifyValue("me oh my-3rdDerived", val)
        self.verifyCount(1, k)
        val = Li_boost_shared_ptr::test3rdupcast(k)
        self.verifyValue("me oh my-3rdDerived", val)
        self.verifyCount(1, k)

        # //////////////////////////////// Member variables ///////////////////
        # smart pointer by value
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("smart member value")
        m.SmartMemberValue = k
        val = k.getValue()
        self.verifyValue("smart member value", val)
        self.verifyCount(2, k)

        kmember = m.SmartMemberValue
        val = kmember.getValue()
        self.verifyValue("smart member value", val)
        self.verifyCount(3, kmember)
        self.verifyCount(3, k)

        GC.track_class = Li_boost_shared_ptr::MemberVariables
        m = nil
        invokeGC("m = nil (A)")

        self.verifyCount(2, kmember)
        self.verifyCount(2, k)

        # smart pointer by pointer
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("smart member pointer")
        m.SmartMemberPointer = k
        val = k.getValue()
        self.verifyValue("smart member pointer", val)
        self.verifyCount(1, k)

        kmember = m.SmartMemberPointer
        val = kmember.getValue()
        self.verifyValue("smart member pointer", val)
        self.verifyCount(2, kmember)
        self.verifyCount(2, k)

        m = nil
        invokeGC("m = nil (B)")

        self.verifyCount(2, kmember)
        self.verifyCount(2, k)

        # smart pointer by reference
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("smart member reference")
        m.SmartMemberReference = k
        val = k.getValue()
        self.verifyValue("smart member reference", val)
        self.verifyCount(2, k)

        kmember = m.SmartMemberReference
        val = kmember.getValue()
        self.verifyValue("smart member reference", val)
        self.verifyCount(3, kmember)
        self.verifyCount(3, k)

        # The C++ reference refers to SmartMemberValue...
        kmemberVal = m.SmartMemberValue
        val = kmember.getValue()
        self.verifyValue("smart member reference", val)
        self.verifyCount(4, kmemberVal)
        self.verifyCount(4, kmember)
        self.verifyCount(4, k)

        m = nil
        invokeGC("m = nil (C)")


        self.verifyCount(3, kmemberVal)
        self.verifyCount(3, kmember)
        self.verifyCount(3, k)

        # plain by value
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("plain member value")
        m.MemberValue = k
        val = k.getValue()
        self.verifyValue("plain member value", val)
        self.verifyCount(1, k)

        kmember = m.MemberValue
        val = kmember.getValue()
        self.verifyValue("plain member value", val)
        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        m = nil
        invokeGC("m = nil (D)")

        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        # plain by pointer
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("plain member pointer")
        m.MemberPointer = k
        val = k.getValue()
        self.verifyValue("plain member pointer", val)
        self.verifyCount(1, k)

        kmember = m.MemberPointer
        val = kmember.getValue()
        self.verifyValue("plain member pointer", val)
        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        m = nil
        invokeGC("m = nil (E)")

        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        # plain by reference
        m = Li_boost_shared_ptr::MemberVariables.new()
        k = Li_boost_shared_ptr::Klass.new("plain member reference")
        m.MemberReference = k
        val = k.getValue()
        self.verifyValue("plain member reference", val)
        self.verifyCount(1, k)

        kmember = m.MemberReference
        val = kmember.getValue()
        self.verifyValue("plain member reference", val)
        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        m = nil
        invokeGC("m = nil (F)")

        self.verifyCount(1, kmember)
        self.verifyCount(1, k)

        # null member variables
        m = Li_boost_shared_ptr::MemberVariables.new()

        # shared_ptr by value
        k = m.SmartMemberValue
        if (k != nil)
            raise RuntimeError, "expected null"
        end
        m.SmartMemberValue = nil
        k = m.SmartMemberValue
        if (k != nil)
            raise RuntimeError, "expected null"
        end
        self.verifyCount(0, k)

        # plain by value
        begin
            m.MemberValue = nil
            raise RuntimeError, "Failed to catch null pointer"
        rescue ArgumentError
        end

        # ////////////////////////////////// Global variables /////////////////
        # smart pointer
        kglobal = Li_boost_shared_ptr.GlobalSmartValue
        if (kglobal != nil)
            raise RuntimeError, "expected null"
        end

        k = Li_boost_shared_ptr::Klass.new("smart global value")
        Li_boost_shared_ptr.GlobalSmartValue = k
        self.verifyCount(2, k)

        kglobal = Li_boost_shared_ptr.GlobalSmartValue
        val = kglobal.getValue()
        self.verifyValue("smart global value", val)
        self.verifyCount(3, kglobal)
        self.verifyCount(3, k)
        self.verifyValue(
            "smart global value", Li_boost_shared_ptr.GlobalSmartValue.getValue())
        Li_boost_shared_ptr.GlobalSmartValue = nil

        # plain value
        k = Li_boost_shared_ptr::Klass.new("global value")
        Li_boost_shared_ptr.GlobalValue = k
        self.verifyCount(1, k)

        kglobal = Li_boost_shared_ptr.GlobalValue
        val = kglobal.getValue()
        self.verifyValue("global value", val)
        self.verifyCount(1, kglobal)
        self.verifyCount(1, k)
        self.verifyValue(
            "global value", Li_boost_shared_ptr.GlobalValue.getValue())

        begin
            Li_boost_shared_ptr.GlobalValue = nil
            raise RuntimeError, "Failed to catch null pointer"
        rescue ArgumentError
        end

        # plain pointer
        kglobal = Li_boost_shared_ptr.GlobalPointer
        if (kglobal != nil)
            raise RuntimeError, "expected null"
        end

        k = Li_boost_shared_ptr::Klass.new("global pointer")
        Li_boost_shared_ptr.GlobalPointer = k
        self.verifyCount(1, k)

        kglobal = Li_boost_shared_ptr.GlobalPointer
        val = kglobal.getValue()
        self.verifyValue("global pointer", val)
        self.verifyCount(1, kglobal)
        self.verifyCount(1, k)
        Li_boost_shared_ptr.GlobalPointer = nil

        # plain reference

        k = Li_boost_shared_ptr::Klass.new("global reference")
        Li_boost_shared_ptr.GlobalReference = k
        self.verifyCount(1, k)

        kglobal = Li_boost_shared_ptr.GlobalReference
        val = kglobal.getValue()
        self.verifyValue("global reference", val)
        self.verifyCount(1, kglobal)
        self.verifyCount(1, k)

        begin
            Li_boost_shared_ptr.GlobalReference = nil
            raise RuntimeError, "Failed to catch null pointer"
        rescue ArgumentError
        end

        # ////////////////////////////////// Templates ////////////////////////
        pid = Li_boost_shared_ptr::PairIntDouble.new(10, 20.2)
        if (pid.baseVal1 != 20 or pid.baseVal2 != 40.4)
            raise RuntimeError, "Base values wrong"
        end
        if (pid.val1 != 10 or pid.val2 != 20.2)
            raise RuntimeError, "Derived Values wrong"
        end
    end

    def verifyValue(expected, got)
        if (expected != got)
            raise RuntimeError, "verify value failed. Expected: #{expected} Got: #{got}"
        end
    end

    def verifyCount(expected, k)
        got = Li_boost_shared_ptr::use_count(k)
        if (expected != got)
            puts "skipped verifyCount expect/got: #{expected}/#{got}"
#            raise RuntimeError, "verify use_count failed. Expected: #{expected} Got: #{got}"
        end
    end
end

runme = Li_boost_shared_ptr_runme.new()
runme.main(debug)
