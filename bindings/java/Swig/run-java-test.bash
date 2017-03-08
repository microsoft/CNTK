#!/bin/bash
SOL_DIR=$1
BUILD_PATH=$2
START_DIR=`pwd`
START_PATH=$PATH
PATH=$PATH:$BUILD_PATH
cd $SOL_DIR/bindings/java/JavaEvalTest
javac -cp $SOL_DIR/bindings/java/SwigProxyClasses/cntk.jar "src/Main.java"
java -Djava.library.path=$BUILD_PATH -classpath "$SOL_DIR/bindings/java/JavaEvalTest/src:/$SOL_DIR/bindings/java/SwigProxyClasses/cntk.jar" Main
PATH=$START_PATH
cd $START_DIR