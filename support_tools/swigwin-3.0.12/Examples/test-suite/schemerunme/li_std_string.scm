; The test string has some non-ascii characters added
; because our guile wrappers had bugs in that area
(define x "hello - æææ")

(if (not (string=? (test-value x) x))
  (begin (error "Error 1") (exit 1)))

(if (not (string=? (test-const-reference x) x))
  (begin (error "Error 2") (exit 1)))

(define y (test-pointer-out))
(test-pointer y)
(define z (test-const-pointer-out))
(test-const-pointer z)

(define a (test-reference-out))
(test-reference a)

;; test global variables
(GlobalString "whee")
(if (not (string=? (GlobalString) "whee"))
  (error "Error 3"))
(if (not (string=? (GlobalString2) "global string 2"))
  (error "Error 4"))

(define struct (new-Structure))

;; MemberString should be a wrapped class
(define scl (Structure-MemberString-get struct))
(if (not (string=? scl ""))
  (error "Error 4.5"))
(Structure-MemberString-set struct "and how")
(if (not (string=? (Structure-MemberString-get struct) "and how"))
  (error "Error 5"))
(if (not (string=? (Structure-MemberString2-get struct) "member string 2"))
  (error "Error 6"))
(Structure-StaticMemberString "static str")
(if (not (string=? (Structure-StaticMemberString) "static str"))
  (error "Error 7"))
(if (not (string=? (Structure-StaticMemberString2) "static member string 2"))
  (error "Error 8"))

;(if (not (string=? (Structure-ConstMemberString-get struct) "const member string"))
;  (error "Error 9"))
(if (not (string=? (Structure-ConstStaticMemberString) "const static member string"))
  (error "Error 10"))

(exit 0)
