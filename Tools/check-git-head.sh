#!/bin/bash
# vim:set expandtab shiftwidth=2 tabstop=2:

errorCount=0

gitTree=HEAD

checkEmptyStdout()
{
  local cmd=$1
  local message=$2
  local output=$(bash -c "$cmd")
  if [ ! -z "$output" ]
  then
    echo "=============================================================================="
    echo "Error: $message:"
    echo "------------------------------------------------------------------------------"
    echo "$output"
    errorCount=$((errorCount + 1))
  fi
  return 0
}

# TODO switch to something without extra command quoting

checkEmptyStdout \
  "git ls-tree -r -t --name-only $gitTree | tr '[:upper:]' '[:lower:]' | sort | uniq --repeated" \
  "git ls-tree: path names that only differ in case:"

checkEmptyStdout \
  "git grep -l \$'\t' $gitTree -- *.cpp *.h *.cu *.bat *.bs | cut -d: -f2-" \
  "files with hard tabs encountered"

checkEmptyStdout \
  "for i in .gitattributes .gitignore .gitmodules LICENSE.md; do test -z \$(git ls-tree --name-only $gitTree \"\$i\") && echo \"\$i\"; done" \
  "Critical file(s) missing"

# TODO line ending checks
# TODO byte order mark and non-ASCII

if [ $errorCount -ne 0 ]
then
  echo "=============================================================================="
  echo FATAL: $errorCount error\(s\)
  exit 1
fi
exit 0 
