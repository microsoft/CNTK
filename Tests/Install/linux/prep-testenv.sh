#!/bin/bash
# TODO nvidia-smi to check availability of GPUs for GPU tests
# TODO not the cleanest
printf 'export PATH=%s\nexport LD_LIBRARY_PATH=%s\nfind $HOME -type d -name cntk\n' "$PATH" "$LD_LIBRARY_PATH" > /home/testuser/test-env.sh
printf "Now do 'su - testuser' then 'source test-env.sh' to set up test env\n"
