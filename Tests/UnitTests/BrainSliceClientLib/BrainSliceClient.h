#pragma once

#include <Windows.h>
#include <FPGAMessage.h>
#include <FPGAResponseFuture.h>

#ifdef BSC_EXPORTS
#define BSC_API //__declspec(dllexport)
#else
#define BSC_API //__declspec(dllimport)
#endif

namespace BrainSlice
{
    using namespace FPGA::LowLevel;

    class BSC_API BrainSliceClient
    {
    public:

        // Use to instantiate
        static BrainSliceClient *Create(bool enableAsync = false);

        // Returns an FPGA message containing a BrainSlice request. 
        //     p_dstIP = IP address of destination FPGA hosting BrainSlice
        //     p_aux = general-purpose arguments that can be sent to BrainSlice program
        //     p_bytes = requested number of bytes for storing and sending a payload to BrainSlice program.
        //     p_data = returned pointer to the requested payload.
        virtual FPGARequest CreateRequest(const uint32_t p_dstIP, std::vector<uint32_t> p_aux, void **p_data, const size_t p_bytes) = 0;

        // Decode FPGA response into BrainSlice response.
        //     p_aux = arguments returning from BrainSlice
        //     p_bytes = number of payload bytes received from BrainSlice
        //     return value = pointer to response payload from BrainSlice
        virtual void * DecodeResponse(FPGAResponse &p_fpgaResponse, std::vector<uint32_t> &p_aux, size_t &p_bytes) = 0;


        // Performs synchronous send to BrainSlice processor.  Caller blocks until response is returned.
        virtual bool SendSync(uint32_t dstIP, FPGARequest &p_fpgaRequest, FPGAResponse &p_fpgaResponse) = 0;

        // Performs asynchronous send to BrainSlice processor.  Requires future call to GetResponse.
        // Requires enableAsync = true when creating BrainSliceClient.
        virtual bool SendAsync(uint32_t dstIP, FPGARequest &p_fpgaRequest, FPGAResponseFuture &p_fpgaResponse) = 0;

        // Blocks until FPGA response is returned. Must be paired with earlier SendAsync.
        // Example: 
        //    FPGAResponseFuture fpgaResponseFuture;
        //    m_bsclient->SendAsync(p_dstIP, fpgaRequest, fpgaResponseFuture);
        //    auto& fpgaResponse = m_bsclient->GetResponse(fpgaResponseFuture);
        virtual FPGAResponse & GetResponse(FPGAResponseFuture &p_fpgaResponseFuture) = 0;


        // Load new BrainSlice program
        //     p_dstIP = IP address of destination IP.
        //     p_dataFile = binary file containing BrainSlice data section
        //     p_instFile = binary file containing BrainSlice code section 
        virtual bool LoadProgram(const uint32_t p_dstIP, const std::string &p_dataFile, const std::string &p_instFile) = 0;


    };
}