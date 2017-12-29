#! /bin/sh

# Bootstrap the development environment - add extra files needed to run configure. 
# Note autoreconf should do what this file achieves, but it has a bug when working with automake!
# The latest config.guess and config.sub should be copied into Tools/config.
# This script will ensure the latest is copied from your autotool installation.

set -e
set -x
test -d Tools/config || mkdir Tools/config
${ACLOCAL-aclocal} -I Tools/config
${AUTOHEADER-autoheader}
${AUTOMAKE-automake} --add-missing --copy --force-missing
${AUTOCONF-autoconf}
cd CCache && ${AUTORECONF-autoreconf}
