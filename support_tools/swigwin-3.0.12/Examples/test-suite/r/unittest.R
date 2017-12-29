unittest <- function (x,y) {
  if (all(x==y)) {
    print("PASS") 
  } else {
    print("FAIL") 
    stop("Test failed")
  }
}

unittesttol <- function(x,y,z) {
  if (all(abs(x-y) < z)) {
    print("PASS")
  } else {
    print("FAIL")
    stop("Test failed")
  }
}
