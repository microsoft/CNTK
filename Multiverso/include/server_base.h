#ifndef _MULTIVERSO_SERVER_BASE_H_
#define _MULTIVERSO_SERVER_BASE_H_

namespace multiverso
{
    class ServerBase
    {
    public:
        virtual void Start() = 0;
        virtual void Stop() = 0;
        virtual bool IsWorking() = 0;
    };
}

#endif  // _MULTIVERSO_SERVER_BASE_H_