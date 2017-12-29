import typedef_typedef

b = typedef_typedef.B()
if b.getValue(123) != 1234:
    raise Exception("Failed")
