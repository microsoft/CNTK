import director_pass_by_value

passByVal = None
class director_pass_by_value_Derived(director_pass_by_value.DirectorPassByValueAbstractBase):
  def virtualMethod(self, b):
    global passByVal
    passByVal = b

# bug was the passByVal global object was destroyed after the call to virtualMethod had finished.
director_pass_by_value.Caller().call_virtualMethod(director_pass_by_value_Derived())
ret = passByVal.getVal();
if ret != 0x12345678:
  raise RuntimeError("Bad return value, got " + hex(ret))
