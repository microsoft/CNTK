#include "BrainSliceClient.h"
#include "BrainSliceIOLib.h"

#include <GraniteFPGALowLevelShell.h>
#include <FPGAMessage.h>
#include <FPGABuffer.h>
#include <FPGAManagementLib.h>

#include <fstream>

namespace BrainSlice
{
    static void assert_bsio(BSIO_STATUS status)
    {
        if (status != BSIO_STATUS_SUCCESS)
        {
            throw new std::runtime_error("BSIO failure, status code: " + status);
        }
    }

    class BrainSliceClientImpl : public BrainSliceClient
    {

    private:
        GraniteFPGALowLevelShell *m_fpga;

    public:

		void ConfigureLargeSlots()
		{
            FPGA_STATUS fpgaStatus = FPGA_STATUS_SUCCESS;
            FPGA_HANDLE fpgaHandle;

			const DWORD
				c_bytesPerSlot = 1024u * 1024u,
				c_numSlots = 64;

            if ((fpgaStatus = FPGA_CreateHandle(&fpgaHandle, 0, FPGA_HANDLE_FLAG_EXCLUSIVE, nullptr, nullptr, nullptr)) != FPGA_STATUS_SUCCESS)
                goto finish;
            
            if ((fpgaStatus = FPGA_SetSlotSize(&fpgaHandle, 1, c_bytesPerSlot, c_numSlots)) != FPGA_STATUS_SUCCESS)
                goto finish;

            if ((fpgaStatus = FPGA_CloseHandle(fpgaHandle)) != FPGA_STATUS_SUCCESS)
                goto finish;

		finish:
            if (fpgaStatus != FPGA_STATUS_SUCCESS)
            {
                throw new std::runtime_error("Failed to configure large slots: " + fpgaStatus);
            }
		}

		
        BrainSliceClientImpl(bool enableAsync = false)
        {
			ConfigureLargeSlots();

            FPGALowLevelConfiguration configuration(FPGALowLevelConfiguration::FPGAMode_hardware);

            configuration.m_asyncMode = enableAsync;

            m_fpga = new GraniteFPGALowLevelShell(configuration);

            if (!m_fpga->Init())
            {
                throw std::runtime_error("Failed to initialize FPGA");
            }
        }

        ~BrainSliceClientImpl()
        {
            delete m_fpga;
        }

        FPGARequest CreateRequest(const uint32_t p_dstIP, std::vector<uint32_t> p_aux, void **p_data, const size_t p_bytes)
        {
            FPGARequest fpgaRequest = m_fpga->InitRequest(0);
            FPGARequestMessage fpgaRequestMessage(fpgaRequest);
            std::vector<uint32_t> hops = { p_dstIP, m_fpga->GetLocalIP() };
            void *payload;

            fpgaRequestMessage.StartMessage((uint8_t)hops.size());
            {
                BSIO_SUBMESSAGE bsioSubMsg;
                DWORD bsioSubMsgSize, *aux;

                assert_bsio(BSIO_CreateSubmessage(&bsioSubMsg, fpgaRequest.GetCurrent(), (DWORD)fpgaRequest.GetRemainingBufferSize()));
                assert_bsio(BSIO_Request_ReserveAux(bsioSubMsg, (DWORD)p_aux.size(), (PVOID *)&aux));

                for (int i = 0; i < p_aux.size(); ++i)
                {
                    aux[i] = p_aux[i];
                }

                assert_bsio(BSIO_Request_ReservePayload(bsioSubMsg, (DWORD)p_bytes, &payload));
                assert_bsio(BSIO_GetSize(bsioSubMsg, &bsioSubMsgSize));
                assert_bsio(BSIO_Request_Finalize(bsioSubMsg));

                fpgaRequest.Reserve(bsioSubMsgSize);
            }
            fpgaRequestMessage.EndMessage(hops);

            if (!p_data)
            {
                throw new std::runtime_error("null pointer provided for p_data");
            }

            *p_data = payload;
            return fpgaRequest;
        }

        void * DecodeResponse(FPGAResponse &p_fpgaResponse, std::vector<uint32_t> &p_aux, size_t &p_bytes)
        {
            BSIO_SUBMESSAGE bsioSubMsg;
            size_t *aux;
            PVOID payload;
            DWORD dwCount,
                payloadBytes;

            FPGAResponseMessage fpgaResponseMessage(p_fpgaResponse);
            SubMessageHeader *msgHeader = nullptr;
            FPGABuffer msgBuffer;

            while ((msgHeader = fpgaResponseMessage.ReadNextSubMessage(msgBuffer)) != nullptr)
            {
                if (msgHeader->m_type == SubMessageType_APPLICATION_SUBMESSAGE)
                {
                    PVOID subMessage = msgBuffer.GetCurrent() - sizeof(SubMessageHeader);
                    assert_bsio(BSIO_CreateSubmessage(&bsioSubMsg, subMessage, (DWORD)msgHeader->m_length));
                    assert_bsio(BSIO_Response_GetAux(bsioSubMsg, (PVOID *)&aux, &dwCount));
                    p_aux.resize(dwCount);
                    for (int i = 0; i < (int)dwCount; ++i)
                    {
                        p_aux[i] = (uint32_t)aux[i];
                    }
                    assert_bsio(BSIO_Response_GetPayload(bsioSubMsg, &payload, &payloadBytes));
                    p_bytes = payloadBytes;
                    break;
                }
            }

            return payload;
        }

        bool SendSync(uint32_t dstIP, FPGARequest &p_fpgaRequest, FPGAResponse &p_fpgaResponse)
        {
            p_fpgaResponse = m_fpga->SyncSendRequest(p_fpgaRequest, dstIP);

            if (p_fpgaResponse.GetStatus() != FPGA_STATUS_SUCCESS)
            {
                throw new std::runtime_error("FPGA error status during SendSync: " + p_fpgaResponse.GetStatus());
            }

            return true;
        }

        bool SendAsync(uint32_t dstIP, FPGARequest &p_fpgaRequest, FPGAResponseFuture &p_fpgaResponse)
        {
            p_fpgaResponse = m_fpga->AsyncSendRequest(p_fpgaRequest, dstIP);

            return true;
        }

        FPGAResponse & GetResponse(FPGAResponseFuture &p_fpgaResponseFuture)
        {
            auto &response = p_fpgaResponseFuture.GetResponse();

            if (response.GetStatus() != FPGA_STATUS_SUCCESS)
            {
                throw new std::runtime_error("FPGA error status on async response: " + response.GetStatus());
            }

            return response;
        }


    private:

        FPGARequest _CreateLoadRequest(uint32_t p_dstIP, void **p_textSection, void **p_dataSection, const size_t textBytes, const size_t dataBytes)
        {
            FPGARequest fpgaRequest = m_fpga->InitRequest(2);
            FPGARequestMessage fpgaRequestMessage(fpgaRequest);
            std::vector<uint32_t> hops = { p_dstIP, m_fpga->GetLocalIP() };
            void *instPtr, *dataPtr;

            fpgaRequestMessage.StartMessage((uint8_t)hops.size());
            {
                BSIO_SUBMESSAGE bsioSubMsg;
                DWORD bsioSubMsgSize;

                assert_bsio(BSIO_CreateSubmessage(&bsioSubMsg, fpgaRequest.GetCurrent(), (DWORD)fpgaRequest.GetRemainingBufferSize()));
                assert_bsio(BSIO_Request_ReserveProgram(bsioSubMsg, (DWORD)textBytes, (DWORD)dataBytes, &instPtr, &dataPtr));

                *p_textSection = instPtr;
                *p_dataSection = dataPtr;

                assert_bsio(BSIO_GetSize(bsioSubMsg, &bsioSubMsgSize));
                assert_bsio(BSIO_Request_Finalize(bsioSubMsg));

                fpgaRequest.Reserve(bsioSubMsgSize);
            }
            fpgaRequestMessage.EndMessage(hops);

            return fpgaRequest;
        }

        bool LoadProgram(const uint32_t p_dstIP, const std::string &p_instFile, const std::string &p_dataFile)
        {
            FPGARequest fpgaRequest(0);
            {
                void *textBuf, *dataBuf;

                std::ifstream dataFile(p_dataFile.c_str(), std::ios::binary);
                if(!dataFile.is_open())
                {
                    throw new std::runtime_error("Unable to open data file: " + p_dataFile);
                }

                std::ifstream instFile(p_instFile.c_str(), std::ios::binary);
                if (!instFile.is_open())
                {
                    throw new std::runtime_error("Unable to open instructions file: " + p_instFile);
                }
				
				dataFile.seekg(0, std::ios::end);
				instFile.seekg(0, std::ios::end);
				size_t instSize = instFile.tellg();
				size_t dataSize = dataFile.tellg();
				dataFile.seekg(0, std::ios::beg);
				instFile.seekg(0, std::ios::beg);

                fpgaRequest = _CreateLoadRequest(p_dstIP, &textBuf, &dataBuf, instSize, dataSize);

                instFile.read((char *)textBuf, instSize);
                dataFile.read((char *)dataBuf, dataSize);
            }

            FPGAResponse fpgaResponse;
            SendSync(p_dstIP, fpgaRequest, fpgaResponse);

            if (fpgaResponse.GetStatus() != FPGA_STATUS_SUCCESS)
            {
                throw new std::runtime_error("FPGA error status received during program load: " + fpgaResponse.GetStatus());
            }

            return true;
        }

    };

    BrainSliceClient * BrainSliceClient::Create(bool enableAsync)
    {
        return new BrainSliceClientImpl(enableAsync);
    }
}