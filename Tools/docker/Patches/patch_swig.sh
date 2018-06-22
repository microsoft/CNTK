#!/bin/bash

# Swig doesn't handle C#/wchar_t conversions well on Linux; for more info, see http://github.com/swig/swig/issues/1233. This script will apply
# a minimal patch to Swig code for use with CNTK. These patches are a minimal set of changes and may not handle all cases.
#
# Patches were created with the command line:
#	diff -u <original filename> <modified filename>

if [[ "$1" == "" ]]
then
   echo "Usage: $0 <swig path>"
   exit -1
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $script_dir/Swig > /dev/null

for patch_filename in $script_dir/Swig/*.patch; do
	filename=$(basename -- "$patch_filename")
	filename=${filename%.patch}
	
	patch --forward $1/csharp/$filename --input $patch_filename
done


popd > /dev/null
