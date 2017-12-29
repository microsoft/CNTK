(test-preassert 1 2)
(expect-throw 'swig-contract-assertion-failed
	      (test-preassert -1 2))
(test-postassert 3)
(expect-throw 'swig-contract-assertion-failed
	      (test-postassert -3))
(test-prepost 2 3)
(test-prepost 5 -4)
(expect-throw 'swig-contract-assertion-failed
	      (test-prepost -3 4))
(expect-throw 'swig-contract-assertion-failed
	      (test-prepost 4 -10))

(exit 0)
