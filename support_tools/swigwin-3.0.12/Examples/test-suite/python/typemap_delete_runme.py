import typemap_delete

r = typemap_delete.Rect(123)
if r.val != 123:
    raise RuntimeError
