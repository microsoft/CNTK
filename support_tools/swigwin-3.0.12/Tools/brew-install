#!/bin/bash

# Wrapper around 'brew install' emitting a message every minute if the command is still running.
# This is used on Travis to ensure the install isn't killed when there is no output over a long period (10 minutes).
# Usage: brew-install package, where package is the name of the package for brew to install.

seconds=0
minutes=0
brew install $1 &
while true; do
  ps -p$! 2>& 1>/dev/null
  if [ $? = 0 ]; then
    if [ $seconds = 60 ]; then
      let seconds=0
      let minutes=minutes+1
      echo "brew install $1 still running ($minutes min)"
    fi
    sleep 1
    let seconds=seconds+1
  else
    break
  fi
done
wait $!
exit $?
