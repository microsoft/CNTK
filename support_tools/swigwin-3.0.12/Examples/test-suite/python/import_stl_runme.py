import import_stl_b
import import_stl_a

v_new = import_stl_b.process_vector([1, 2, 3])
if v_new != (1, 2, 3, 4):
    raise RuntimeError, v_new
