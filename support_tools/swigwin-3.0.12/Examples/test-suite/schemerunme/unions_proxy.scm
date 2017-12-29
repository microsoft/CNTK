;;; This is the union runtime testcase. It ensures that values within a
;;; union embedded within a struct can be set and read correctly.

;; Create new instances of SmallStruct and BigStruct for later use
(define small (make <SmallStruct>))
(slot-set! small 'jill 200)

(define big (make <BigStruct>))
(slot-set! big 'smallstruct small)
(slot-set! big 'jack 300)

;; Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
;; Ensure values in EmbeddedUnionTest are set correctly for each.
(define eut (make <EmbeddedUnionTest>))

;; First check the SmallStruct in EmbeddedUnionTest
(slot-set! eut 'number 1)
(slot-set! (slot-ref eut 'uni) 'small small)
(let ((Jill1 (slot-ref
	       (slot-ref 
		  (slot-ref eut 'uni) 
		  'small)
	       'jill)))
  (if (not (= Jill1 200))
      (begin
	(display "Runtime test 1 failed.")
	(exit 1))))

(let ((Num1 (slot-ref eut 'number)))
  (if (not (= Num1 1))
      (begin
	(display "Runtime test 2 failed.")
	(exit 1))))

;; that should do

(exit 0)
