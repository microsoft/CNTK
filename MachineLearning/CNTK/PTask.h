//
// <copyright file="PTask.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// A single include file to collect together all includes for PTask.
#define CUDA_SUPPORT
#include "PTaskAPI.h"
#include "primitive_types.h"
#include "HostTask.h"

using namespace PTask;

// TODO: Base the path on the properties specified in config.txt. 
#define PTASK_GRAPH_VIZ_FILE "C:\\temp\\PTaskGraph.dot"
