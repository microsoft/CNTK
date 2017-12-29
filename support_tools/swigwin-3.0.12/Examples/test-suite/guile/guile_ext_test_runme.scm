(dynamic-call "scm_init_guile_ext_test_module" (dynamic-link "./libguile_ext_test"))

; This is a test for SF Bug 1573892
; If IsPointer is called before TypeQuery, the test-is-pointer will fail
; (i.e if the bottom two lines were moved to the top, the old code would succeed)
; only a problem when is-pointer is called first

(define a (new-A))

(if (not (test-is-pointer a))
  (error "test-is-pointer failed!"))

(if (test-is-pointer 5)
  (error "test-is-pointer thinks 5 is a pointer!"))

(define b (test-create))
(A-hello b)

(exit 0)
