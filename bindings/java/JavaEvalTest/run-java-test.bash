#!/bin/bash
set -e -o pipefail

CNTK_PATH="$1"
LIB_PATH="$2"
cd $CNTK_PATH/bindings/java/JavaEvalTest
javac -cp $CNTK_PATH/bindings/java/Swig src/Main.java
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
java -classpath "$CNTK_PATH/bindings/java/JavaEvalTest/src:$CNTK_PATH/bindings/java/Swig" Main
