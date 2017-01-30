//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>

#ifdef _WIN32

#include <WinSock.h> // Note: this may conflict with WinSock2.h users (dup definition errors; don't know a general solution)

#pragma comment(lib, "ws2_32.lib")

#else
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// GetHostName() -- function (disguised as a class) to get the machine name
// usage:
//    std::string hostname = GetHostname();
// ---------------------------------------------------------------------------

class GetHostName : public std::string
{
public:
    GetHostName()
    {
        static std::string hostname; // it's costly, so we cache the name
        if (hostname.empty())
        {
#ifdef _WIN32
            WSADATA wsaData;
            if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
            {
                return;
            }
#endif
            char hostnamebuf[1024];
            strcpy_s(hostnamebuf, 1024, "localhost"); // in case it goes wrong
            ::gethostname(&hostnamebuf[0], sizeof(hostnamebuf) / sizeof(*hostnamebuf));
            hostname = hostnamebuf;
#ifdef _WIN32
            WSACleanup();
#endif
        }
        assign(hostname);
    }
};
