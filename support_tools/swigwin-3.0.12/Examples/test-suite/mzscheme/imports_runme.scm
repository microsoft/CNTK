;;; This file is part of a test for SF bug #231619. 
;;; It shows that the %import directive does not work properly in SWIG
;;; 1.3a5:  Type information is not properly generated if a base class
;;; comes from an %import-ed file. 

(load-extension "imports_a.so")
(load-extension "imports_b.so")

(define x (new-B))

;; This fails in 1.3a5 because the SWIG runtime code does not know
;; that x (an instance of class B) can be passed to methods of class A. 

(A-hello x)				

(exit 0)
