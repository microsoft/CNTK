#!/usr/bin/env python
"""
From SWIG 1.3.37 we deprecated all SWIG symbols that start with Py,
since they are inappropriate and discouraged in Python documentation
(from http://www.python.org/doc/2.5.2/api/includes.html):

"All user visible names defined by Python.h (except those defined by the included
standard headers) have one of the prefixes "Py" or "_Py". Names beginning with
"_Py" are for internal use by the Python implementation and should not be used
by extension writers. Structure member names do not have a reserved prefix.

Important: user code should never define names that begin with "Py" or "_Py".
This confuses the reader, and jeopardizes the portability of the user code to
future Python versions, which may define additional names beginning with one
of these prefixes."

This file is a simple script used for change all of these symbols, for user code
or SWIG itself. 
"""
import re
from shutil import copyfile
import sys

symbols = [
        #(old name, new name)
        ("PySequence_Base", "SwigPySequence_Base"),
        ("PySequence_Cont", "SwigPySequence_Cont"),
        ("PySwigIterator_T", "SwigPyIterator_T"),
        ("PyPairBoolOutputIterator", "SwigPyPairBoolOutputIterator"),
        ("PySwigIterator", "SwigPyIterator"),
        ("PySwigIterator_T", "SwigPyIterator_T"),
        ("PyMapIterator_T", "SwigPyMapIterator_T"),
        ("PyMapKeyIterator_T", "SwigPyMapKeyIterator_T"),
        ("PyMapValueIterator_T", "SwigPyMapValueITerator_T"),
        ("PyObject_ptr", "SwigPtr_PyObject"),
        ("PyObject_var", "SwigVar_PyObject"),
        ("PyOper", "SwigPyOper"),
        ("PySeq", "SwigPySeq"),
        ("PySequence_ArrowProxy", "SwigPySequence_ArrowProxy"),
        ("PySequence_Cont", "SwigPySequence_Cont"),
        ("PySequence_InputIterator", "SwigPySequence_InputIterator"),
        ("PySequence_Ref", "SwigPySequence_Ref"),
        ("PySwigClientData", "SwigPyClientData"),
        ("PySwigClientData_Del", "SwigPyClientData_Del"),
        ("PySwigClientData_New", "SwigPyClientData_New"),
        ("PySwigIterator", "SwigPyIterator"),
        ("PySwigIteratorClosed_T", "SwigPyIteratorClosed_T"),
        ("PySwigIteratorOpen_T", "SwigPyIteratorOpen_T"),
        ("PySwigIterator_T", "SwigPyIterator_T"),
        ("PySwigObject", "SwigPyObject"),
        ("PySwigObject_Check", "SwigPyObject_Check"),
        ("PySwigObject_GetDesc", "SwigPyObject_GetDesc"),
        ("PySwigObject_New", "SwigPyObject_New"),
        ("PySwigObject_acquire", "SwigPyObject_acquire"),
        ("PySwigObject_append", "SwigPyObject_append"),
        ("PySwigObject_as_number", "SwigPyObject_as_number"),
        ("PySwigObject_compare", "SwigPyObject_compare"),
        ("PySwigObject_dealloc", "SwigPyObject_dealloc"),
        ("PySwigObject_disown", "SwigPyObject_disown"),
        ("PySwigObject_format", "SwigPyObject_format"),
        ("PySwigObject_getattr", "SwigPyObject_getattr"),
        ("PySwigObject_hex", "SwigPyObject_hex"),
        ("PySwigObject_long", "SwigPyObject_long"),
        ("PySwigObject_next", "SwigPyObject_next"),
        ("PySwigObject_oct", "SwigPyObject_oct"),
        ("PySwigObject_own", "SwigPyObject_own"),
        ("PySwigObject_print", "SwigPyObject_print"),
        ("PySwigObject_repr", "SwigPyObject_repr"),
        ("PySwigObject_richcompare", "SwigPyObject_richcompare"),
        ("PySwigObject_str", "SwigPyObject_str"),
        ("PySwigObject_type", "SwigPyObject_type"),
        ("PySwigPacked", "SwigPyPacked"),
        ("PySwigPacked_Check", "SwigPyPacked_Check"),
        ("PySwigPacked_New", "SwigPyPacked_New"),
        ("PySwigPacked_UnpackData", "SwigPyPacked_UnpackData"),
        ("PySwigPacked_compare", "SwigPyPacked_compare"),
        ("PySwigPacked_dealloc", "SwigPyPacked_dealloc"),
        ("PySwigPacked_print", "SwigPyPacked_print"),
        ("PySwigPacked_repr", "SwigPyPacked_repr"),
        ("PySwigPacked_str", "SwigPyPacked_str"),
        ("PySwigPacked_type", "SwigPyPacked_type"),
        ("pyseq", "swigpyseq"),
        ("pyswigobject_type", "swigpyobject_type"),
        ("pyswigpacked_type", "swigpypacked_type"),
    ]

res = [(re.compile("\\b(%s)\\b"%oldname), newname) for oldname, newname in symbols]

def patch_file(fn):
    newf = []
    changed = False
    for line in open(fn):
        for r, newname in res:
            line, n = r.subn(newname, line)
            if n>0:
                changed = True
        newf.append(line)

    if changed:
        copyfile(fn, fn+".bak")
        f = open(fn, "w")
        f.write("".join(newf))
        f.close()
    return changed

def main(fns):
    for fn in fns:
        try:
            if patch_file(fn):
                print "Patched file", fn
        except IOError:
            print "Error occurred during patching", fn
    return

if __name__=="__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print "Patch your interface file for SWIG's Py* symbol name deprecation."
        print "Usage:"
        print "    %s files..."%sys.argv[0]

        
