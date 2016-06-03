#pragma once

#include <string>
#include <ctime>
#include <functional>
#include <iostream>


std::string FormatTime(time_t tm);

void PrintTime(std::function<void()> fn, const std::string& tag);

float GetTime(); 
