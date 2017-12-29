#!/bin/sh

pcre_subdir=pcre/pcre-swig-install
pcre_install_dir=`pwd`/$pcre_subdir

usage() {
  echo "Helper script to build PCRE as a static library from a tarball just for use during the"
  echo "SWIG build. It does not install PCRE for global use on your system."
  echo "Usage: pcre-build.sh [--help] [args]"
  echo "  args   - optional additional arguments passed on to the PCRE configure script (leave out"
  echo "         unless you are an expert at configure)"
  echo "  --help - Display this help information."
  echo "Instructions:"
  echo "  - Download the latest PCRE source tarball from http://www.pcre.org and place in the"
  echo "    directory that you will configure and build SWIG."
  echo "  - Run this script in the same directory that you intend to configure and build SWIG in."
  echo "    This will configure and build PCRE as a static library."
  echo "  - Afterwards run the SWIG configure script which will then find and use the PCRE static"
  echo "    libraries in the $pcre_subdir subdirectory."
  exit 0
}

bail() {
  echo $1 >&2
  exit 1
}

if test "$1" = "-h" -o "$1" = "-help" -o "$1" = "--help" ; then
  usage
fi

if test -f "pcre-build.sh" ; then
  echo "Error: this script should not be run in the Tools directory" >&2
  echo ""
  usage
fi

echo "Looking for PCRE tarball..."
rm -rf pcre
pcre_tarball=`ls pcre-*.tar*`
test -n "$pcre_tarball" || bail "Could not find tarball matching pattern: pcre-*.tar*"
test -f "$pcre_tarball" || bail "Could not find a single PCRE tarball. Found: $pcre_tarball"

echo "Extracting tarball: $pcre_tarball"
tar -xf $pcre_tarball || bail "Could not untar $pcre_tarball"
pcre_dir=`echo $pcre_tarball | sed -e "s/\.tar.*//"`
echo "Configuring PCRE in directory: pcre"
mv $pcre_dir pcre || bail "Could not create pcre directory"
cd pcre && ./configure --prefix=$pcre_install_dir --disable-shared $* || bail "PCRE configure failed"
echo "Building PCRE..."
${MAKE:-make} -s || bail "Could not build PCRE"
echo "Installing PCRE locally to $pcre_install_dir..."
${MAKE:-make} -s install || bail "Could not install PCRE"
echo ""
echo "The SWIG configure script can now be run, whereupon PCRE will automatically be detected and used from $pcre_install_dir/bin/pcre-config."
