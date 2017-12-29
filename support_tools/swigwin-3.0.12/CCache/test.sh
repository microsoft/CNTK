#!/bin/sh

# a simple test suite for ccache
# tridge@samba.org

if test -n "$CC"; then
 COMPILER="$CC"
else
 COMPILER=cc
fi

if test -n "$SWIG"; then
 SWIG="$SWIG"
else
 SWIG=swig
fi

# fix: Remove ccache from $PATH if it exists
#      as it will influence the unit tests
PATH="`echo $PATH | \
 sed -e 's!:/usr\(/local\)*/lib\([0-9]\)*/ccache\(/\)*!!g'`"

CCACHE=../ccache-swig
TESTDIR=test.$$

test_failed() {
    reason="$1"
    echo $1
    $CCACHE -s
    cd ..
    rm -rf $TESTDIR
    echo TEST FAILED
    exit 1
}

randcode() {
    outfile="$1"
    nlines=$2
    i=0;
    (
    while [ $i -lt $nlines ]; do
	echo "int foo$nlines$i(int x) { return x; }"
	i=`expr $i + 1`
    done
    ) >> "$outfile"
}

genswigcode() {
    outfile="$1"
    nlines=$2
    i=0;
    (
    echo "%module swigtest$2;"
    while [ $i -lt $nlines ]; do
        echo "int foo$nlines$i(int x);"
        echo "struct Bar$nlines$i { int y; };"
        i=`expr $i + 1`
    done
    ) >> "$outfile"
}


getstat() {
    stat="$1"
    value=`$CCACHE -s | grep "$stat" | cut -c34-40`
    echo $value
}

checkstat() {
    stat="$1"
    expected_value="$2"
    value=`getstat "$stat"`
#    echo "exp: $expected_value got: $value $testname"
    if [ "$expected_value" != "$value" ]; then
	test_failed "SUITE: $testsuite TEST: $testname - Expected $stat to be $expected_value got $value"
    fi
}


basetests() {
    echo "starting testsuite $testsuite"
    rm -rf "$CCACHE_DIR"
    checkstat 'cache hit' 0
    checkstat 'cache miss' 0

    j=1
    rm -f *.c
    while [ $j -lt 32 ]; do
	randcode test$j.c $j
	j=`expr $j + 1`
    done

    testname="BASIC"
    $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 0
    checkstat 'cache miss' 1
    
    testname="BASIC2"
    $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 1
    checkstat 'cache miss' 1
    
    testname="debug"
    $CCACHE_COMPILE -c test1.c -g
    checkstat 'cache hit' 1
    checkstat 'cache miss' 2
    
    testname="debug2"
    $CCACHE_COMPILE -c test1.c -g
    checkstat 'cache hit' 2
    checkstat 'cache miss' 2
    
    testname="output"
    $CCACHE_COMPILE -c test1.c -o foo.o
    checkstat 'cache hit' 3
    checkstat 'cache miss' 2

    testname="link"
    $CCACHE_COMPILE test1.c -o test 2> /dev/null
    checkstat 'called for link' 1

    testname="multiple"
    $CCACHE_COMPILE -c test1.c test2.c
    checkstat 'multiple source files' 1

    testname="find"
    $CCACHE blahblah -c test1.c 2> /dev/null
    checkstat "couldn't find the compiler" 1 

    testname="bad"
    $CCACHE_COMPILE -c test1.c -I 2> /dev/null
    checkstat 'bad compiler arguments' 1

    testname="c/c++"
    ln -f test1.c test1.ccc
    $CCACHE_COMPILE -c test1.ccc 2> /dev/null
    checkstat 'not a C/C++ file' 1

    testname="unsupported"
    $CCACHE_COMPILE -M foo -c test1.c > /dev/null 2>&1
    checkstat 'unsupported compiler option' 1

    testname="stdout"
    $CCACHE echo foo -c test1.c > /dev/null
    checkstat 'compiler produced stdout' 1

    testname="non-regular"
    mkdir testd
    $CCACHE_COMPILE -o testd -c test1.c > /dev/null 2>&1
    rm -rf testd
    checkstat 'output to a non-regular file' 1

    testname="no-input"
    $CCACHE_COMPILE -c -O2 2> /dev/null
    checkstat 'no input file' 1


    testname="CCACHE_DISABLE"
    CCACHE_DISABLE=1 $CCACHE_COMPILE -c test1.c 2> /dev/null
    checkstat 'cache hit' 3 
    $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 4 

    testname="CCACHE_CPP2"
    CCACHE_CPP2=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 4 
    checkstat 'cache miss' 3

    CCACHE_CPP2=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 5 
    checkstat 'cache miss' 3

    testname="CCACHE_NOSTATS"
    CCACHE_NOSTATS=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 5
    checkstat 'cache miss' 3
    
    testname="CCACHE_RECACHE"
    CCACHE_RECACHE=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 5 
    checkstat 'cache miss' 4

    # strictly speaking should be 6 - RECACHE causes a double counting!
    checkstat 'files in cache' 8 
    $CCACHE -c > /dev/null
    checkstat 'files in cache' 6


    testname="CCACHE_HASHDIR"
    CCACHE_HASHDIR=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 5
    checkstat 'cache miss' 5

    CCACHE_HASHDIR=1 $CCACHE_COMPILE -c test1.c -O -O
    checkstat 'cache hit' 6
    checkstat 'cache miss' 5

    checkstat 'files in cache' 8
    
    testname="comments"
    echo '/* a silly comment */' > test1-comment.c
    cat test1.c >> test1-comment.c
    $CCACHE_COMPILE -c test1-comment.c
    rm -f test1-comment*
    checkstat 'cache hit' 6
    checkstat 'cache miss' 6

    testname="CCACHE_UNIFY"
    CCACHE_UNIFY=1 $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 6
    checkstat 'cache miss' 7
    mv test1.c test1-saved.c
    echo '/* another comment */' > test1.c
    cat test1-saved.c >> test1.c
    CCACHE_UNIFY=1 $CCACHE_COMPILE -c test1.c
    mv test1-saved.c test1.c
    checkstat 'cache hit' 7
    checkstat 'cache miss' 7

    testname="cache-size"
    for f in *.c; do
	$CCACHE_COMPILE -c $f
    done
    checkstat 'cache hit' 8
    checkstat 'cache miss' 37
    checkstat 'files in cache' 72
    $CCACHE -F 48 -c > /dev/null
    if [ `getstat 'files in cache'` -gt 48 ]; then
	test_failed '-F test failed'
    fi

    testname="cpp call"
    $CCACHE_COMPILE -c test1.c -E > test1.i
    checkstat 'cache hit' 8
    checkstat 'cache miss' 37

    testname="direct .i compile"
    $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 8
    checkstat 'cache miss' 38

    $CCACHE_COMPILE -c test1.i
    checkstat 'cache hit' 9
    checkstat 'cache miss' 38

    $CCACHE_COMPILE -c test1.i
    checkstat 'cache hit' 10
    checkstat 'cache miss' 38

    # removed these tests as some compilers (including newer versions of gcc)
    # determine which language to use based on .ii/.i extension, and C++ may 
    # not be installed
#     testname="direct .ii file"
#     mv test1.i test1.ii
#     $CCACHE_COMPILE -c test1.ii
#     checkstat 'cache hit' 10
#     checkstat 'cache miss' 39

#     $CCACHE_COMPILE -c test1.ii
#     checkstat 'cache hit' 11
#     checkstat 'cache miss' 39
    
    testname="stripc" # This test might not be portable
    CCACHE_STRIPC=1 $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 10
    checkstat 'cache miss' 39

    CCACHE_STRIPC=1 $CCACHE_COMPILE -c test1.c
    checkstat 'cache hit' 11
    checkstat 'cache miss' 39

    testname="zero-stats"
    $CCACHE -z > /dev/null
    checkstat 'cache hit' 0
    checkstat 'cache miss' 0

    testname="clear"
    $CCACHE -C > /dev/null
    checkstat 'files in cache' 0


    rm -f test1.c
}

swigtests() {
    echo "starting swig testsuite $testsuite"
    rm -rf "$CCACHE_DIR"
    checkstat 'cache hit' 0
    checkstat 'cache miss' 0

    j=1
    rm -f *.i
    genswigcode testswig1.i 1

    testname="BASIC"
    $CCACHE_COMPILE -java testswig1.i
    checkstat 'cache hit' 0
    checkstat 'cache miss' 1
    
    checkstat 'files in cache' 6

    testname="BASIC2"
    $CCACHE_COMPILE -java testswig1.i
    checkstat 'cache hit' 1
    checkstat 'cache miss' 1
    
    testname="output"
    $CCACHE_COMPILE -java testswig1.i -o foo_wrap.c
    checkstat 'cache hit' 1
    checkstat 'cache miss' 2

    testname="bad"
    $CCACHE_COMPILE -java testswig1.i -I 2> /dev/null
    checkstat 'bad compiler arguments' 1

    testname="stdout"
    $CCACHE_COMPILE -v -java testswig1.i > /dev/null
    checkstat 'compiler produced stdout' 1

    testname="non-regular"
    mkdir testd
    $CCACHE_COMPILE -o testd -java testswig1.i > /dev/null 2>&1
    rm -rf testd
    checkstat 'output to a non-regular file' 1

    testname="no-input"
    $CCACHE_COMPILE -java 2> /dev/null
    checkstat 'no input file' 1


    testname="CCACHE_DISABLE"
    CCACHE_DISABLE=1 $CCACHE_COMPILE -java testswig1.i 2> /dev/null
    checkstat 'cache hit' 1 
    $CCACHE_COMPILE -java testswig1.i
    checkstat 'cache hit' 2 

    testname="CCACHE_CPP2"
    CCACHE_CPP2=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 2 
    checkstat 'cache miss' 3

    CCACHE_CPP2=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 3 
    checkstat 'cache miss' 3

    testname="CCACHE_NOSTATS"
    CCACHE_NOSTATS=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 3
    checkstat 'cache miss' 3
    
    testname="CCACHE_RECACHE"
    CCACHE_RECACHE=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 3 
    checkstat 'cache miss' 4

    # strictly speaking should be 3x6=18 instead of 4x6=24 - RECACHE causes a double counting!
    checkstat 'files in cache' 24 
    $CCACHE -c > /dev/null
    checkstat 'files in cache' 18


    testname="CCACHE_HASHDIR"
    CCACHE_HASHDIR=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 3
    checkstat 'cache miss' 5

    CCACHE_HASHDIR=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 4
    checkstat 'cache miss' 5

    checkstat 'files in cache' 24
    
    testname="cpp call"
    $CCACHE_COMPILE -java -E testswig1.i > testswig1-preproc.i
    checkstat 'cache hit' 4
    checkstat 'cache miss' 5

    testname="direct .i compile"
    $CCACHE_COMPILE -java testswig1.i
    checkstat 'cache hit' 5
    checkstat 'cache miss' 5

    # No cache hit due to different input file name, -nopreprocess should not be given twice to SWIG
    $CCACHE_COMPILE -java -nopreprocess testswig1-preproc.i
    checkstat 'cache hit' 5
    checkstat 'cache miss' 6

    $CCACHE_COMPILE -java -nopreprocess testswig1-preproc.i
    checkstat 'cache hit' 6
    checkstat 'cache miss' 6

    testname="stripc"
    CCACHE_STRIPC=1 $CCACHE_COMPILE -java -O -O testswig1.i
    checkstat 'cache hit' 7
    checkstat 'cache miss' 6

    CCACHE_STRIPC=1 $CCACHE_COMPILE -java -O -O -O testswig1.i
    checkstat 'cache hit' 7
    checkstat 'cache miss' 7

    rm -f testswig1-preproc.i
    rm -f testswig1.i
}

######
# main program
rm -rf $TESTDIR
mkdir $TESTDIR
cd $TESTDIR || exit 1

unset CCACHE_DIR
unset CCACHE_TEMPDIR
unset CCACHE_LOGFILE
unset CCACHE_VERBOSE
unset CCACHE_PATH
unset CCACHE_CC
unset CCACHE_PREFIX
unset CCACHE_DISABLE
unset CCACHE_READONLY
unset CCACHE_CPP2
unset CCACHE_NOCOMPRESS
unset CCACHE_NOSTATS
unset CCACHE_NLEVELS
unset CCACHE_HARDLINK
unset CCACHE_RECACHE
unset CCACHE_UMASK
unset CCACHE_HASHDIR
unset CCACHE_UNIFY
unset CCACHE_EXTENSION
unset CCACHE_STRIPC
unset CCACHE_SWIG

CCACHE_DIR="ccache dir" # with space in directory name (like Windows default)
mkdir "$CCACHE_DIR"
export CCACHE_DIR

testsuite="base"
CCACHE_COMPILE="$CCACHE $COMPILER"
basetests
CCACHE_COMPILE="$CCACHE $SWIG"
swigtests

if test -z "$NOSOFTLINKSTEST"; then
  testsuite="link"
  ln -s $CCACHE $COMPILER
  CCACHE_COMPILE="./$COMPILER"
  basetests
  rm "./$COMPILER"
  ln -s $CCACHE $SWIG
  CCACHE_COMPILE="./$SWIG"
  swigtests
  rm "./$SWIG"
else
  echo "skipping testsuite link"
fi

testsuite="hardlink"
CCACHE_COMPILE="env CCACHE_NOCOMPRESS=1 CCACHE_HARDLINK=1 $CCACHE $COMPILER"
basetests
CCACHE_COMPILE="env CCACHE_NOCOMPRESS=1 CCACHE_HARDLINK=1 $CCACHE $SWIG"
swigtests

testsuite="cpp2"
CCACHE_COMPILE="env CCACHE_CPP2=1 $CCACHE $COMPILER"
basetests
CCACHE_COMPILE="env CCACHE_CPP2=1 $CCACHE $SWIG"
swigtests

testsuite="nlevels4"
CCACHE_COMPILE="env CCACHE_NLEVELS=4 $CCACHE $COMPILER"
basetests

testsuite="nlevels1"
CCACHE_COMPILE="env CCACHE_NLEVELS=1 $CCACHE $COMPILER"
basetests

cd ..
rm -rf $TESTDIR
echo test done - OK
exit 0
