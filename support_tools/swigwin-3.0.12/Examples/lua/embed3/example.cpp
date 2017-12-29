/* File : example.cpp */

#include <stdio.h>
#include "example.h"

void Engine::start()
{
    printf("[C++] Engine::start()\n");
}

void Engine::stop()
{
    printf("[C++] Engine::stop()\n");
}

void Engine::accelerate(float f)
{
    printf("[C++] Engine::accelerate(%f)\n",f);
}

void Engine::decelerate(float f)
{
    printf("[C++] Engine::decelerate(%f)\n",f);
}

