import python_extranative

vs = python_extranative.make_vector_string()
if not isinstance(vs, python_extranative.VectorString):
    # will be of type tuple if extranative not working
    raise RuntimeError("Not of type VectorString")

for s1, s2 in zip(vs, ["one", "two"]):
    if s1 != s2:
        raise RuntimeError("Mismatch: " + s1 + " " + s2)

