#!/bin/bash
#
# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# Log steps, stop on error
set -x -e -o pipefail

USAGE="Usage: $0 <url-or-file-of-cntk-drop>"

DROP_LOCATION=${1?$USAGE}

if [ -f "$DROP_LOCATION" ]; then
  DROP_FILE="$DROP_LOCATION"
else
  # TODO tune
  sudo apt-get install wget ca-certificates
  # Not found locally, assume it's a URL
  wget "$DROP_LOCATION"
  DROP_FILE="$(basename "$DROP_LOCATION")"
fi

tar -xzf "$DROP_FILE"
test -d cntk

exec cntk/Scripts/linux/install-cntk.sh

# vim:set expandtab shiftwidth=2 tabstop=2:
