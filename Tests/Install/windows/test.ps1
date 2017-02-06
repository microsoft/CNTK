# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

$image = 'cntk:installtest'

docker build -t $image .
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
docker run --rm $image powershell c:/local/test-install.ps1
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
docker rmi $image
# Ignore error here
