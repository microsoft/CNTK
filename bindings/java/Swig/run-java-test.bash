#!/bin/bash
set -e -o pipefail

SOL_DIR="$1"
BUILD_PATH="$2"
cd $SOL_DIR/bindings/java/JavaEvalTest
javac -cp $SOL_DIR/bindings/java/SwigProxyClasses/cntk.jar "src/Main.java"
java -Djava.library.path=$BUILD_PATH -classpath "$SOL_DIR/bindings/java/JavaEvalTest/src:/$SOL_DIR/bindings/java/SwigProxyClasses/cntk.jar" Main
