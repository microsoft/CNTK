// simple emulation of spglog.h for things needed by Marian logging.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdio>

namespace spdlog
{
    struct logger
    {
        template <class... Args> void critical(std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
        template <class... Args> void trace   (std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
        template <class... Args> void debug   (std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
        template <class... Args> void info    (std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
        template <class... Args> void warn    (std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
        template <class... Args> void error   (std::string msg, Args...) { fprintf(stderr, "%s\n", msg.c_str()), fflush(stderr); }
    };
    static inline std::shared_ptr<logger> get(const std::string&)
    {
        static std::shared_ptr<logger> us = std::make_shared<logger>();
        return us;
    }
};
