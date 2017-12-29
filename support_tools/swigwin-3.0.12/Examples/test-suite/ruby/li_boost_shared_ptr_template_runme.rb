require 'li_boost_shared_ptr_template'

begin
  b = Li_boost_shared_ptr_template::BaseINTEGER.new()
  d = Li_boost_shared_ptr_template::DerivedINTEGER.new()
  if (b.bar() != 1)
    raise RuntimeError("test 1")
  end
  if (d.bar() != 2)
    raise RuntimeError("test 2")
  end
  if (Li_boost_shared_ptr_template.bar_getter(b) != 1)
    raise RuntimeError("test 3")
  end
# Needs fixing as it does for Python
#  if (Li_boost_shared_ptr_template.bar_getter(d) != 2)
#    raise RuntimeError("test 4")
#  end
end

begin
  b = Li_boost_shared_ptr_template::BaseDefaultInt.new()
  d = Li_boost_shared_ptr_template::DerivedDefaultInt.new()
  d2 = Li_boost_shared_ptr_template::DerivedDefaultInt2.new()
  if (b.bar2() != 3)
    raise RuntimeError("test 5")
  end
  if (d.bar2() != 4)
    raise RuntimeError("test 6")
  end
  if (d2.bar2() != 4)
    raise RuntimeError("test 6")
  end
  if (Li_boost_shared_ptr_template.bar2_getter(b) != 3)
    raise RuntimeError("test 7")
  end
# Needs fixing as it does for Python
#  if (Li_boost_shared_ptr_template.bar2_getter(d) != 4)
#    raise RuntimeError("test 8")
#  end
#  if (Li_boost_shared_ptr_template.bar2_getter(d2) != 4)
#    raise RuntimeError("test 8")
#  end
end

