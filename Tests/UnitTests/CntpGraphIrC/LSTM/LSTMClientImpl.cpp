#include "LSTMClient.h"
#include "BrainSliceClient.h"
#include <FPGAMessage.h>
#include <FPGABuffer.h>

namespace LSTM
{
	using std::vector;
    using namespace BrainSlice;

	class LSTMClientImpl : public LSTMClient
	{
	private:

		// hardware constants
		const uint32_t
			c_nativeDim = 252;

		// brainslice function IDs
		const uint32_t
			c_EvalFunc = 2,
			c_LoadMatFunc = 8,
			c_LoadBiasFunc = 9,
            c_LoopbackFunc = 4;

		// model constants 
		const uint32_t
			c_numBiasVectors = 4,
			c_inputDim = 200,
			c_outputDim = 500;

		BrainSliceClient *m_bsclient;

		void _LoadBiasVectors(const uint32_t p_dstIP, std::vector<std::vector<dword_t> > &p_parameters)
		{
			FPGARequest fpgaRequest(1);

			size_t numPhysVectors = (c_outputDim + c_nativeDim - 1) / c_nativeDim;
			size_t payloadBytes = c_numBiasVectors * numPhysVectors * c_nativeDim * sizeof(dword_t);
			vector<uint32_t> aux = { (uint32_t)c_LoadBiasFunc };
			char *payload;

			fpgaRequest = m_bsclient->CreateRequest(p_dstIP, aux, (void **)&payload, payloadBytes);

			memset(payload, 0x0, payloadBytes);
			for (int i = 0; i < 4; ++i)
			{
				assert(p_parameters[i].size() == c_outputDim);
				memcpy(
					payload + i * numPhysVectors * c_nativeDim * sizeof(dword_t),
					(char *)&p_parameters[i][0],
					c_outputDim * sizeof(dword_t));
			}

			FPGAResponse fpgaResponse;
			m_bsclient->SendSync(p_dstIP, fpgaRequest, fpgaResponse);
		}


		void _LoadWeightMatrix(const uint32_t p_dstIP, std::vector<dword_t> &p_weightMatrix, const size_t rows, const size_t cols, const size_t matAddr)
		{
			FPGARequest fpgaRequest(1);

			assert((rows > 0) && (cols > 0) && (p_weightMatrix.size() == rows * cols));

			size_t numBlockRows = (rows + c_nativeDim - 1) / c_nativeDim;
			size_t numBlockCols = (cols + c_nativeDim - 1) / c_nativeDim;
			size_t numNativeMatrices = numBlockRows * numBlockCols;

			size_t payloadBytes = numBlockRows * numBlockCols * c_nativeDim * c_nativeDim * sizeof(dword_t);
			vector<uint32_t> aux = { (uint32_t)c_LoadMatFunc, (uint32_t)numNativeMatrices, (uint32_t)matAddr };
			dword_t *payload;

			fpgaRequest = m_bsclient->CreateRequest(p_dstIP, aux, (void **)&payload, payloadBytes);

			memset((void *)payload, 0x0, payloadBytes);

			for (size_t r = 0; r < rows; ++r)
			{
				for (size_t c = 0; c < cols; ++c)
				{
					size_t blockRow = r / c_nativeDim;
					size_t blockCol = c / c_nativeDim;
					dword_t val = p_weightMatrix[r * cols + c];

					size_t localRow = r % c_nativeDim;
					size_t localCol = c % c_nativeDim;

					size_t blockOffset = (blockRow * numBlockCols + blockCol) * c_nativeDim * c_nativeDim;
					size_t localOffset = localRow * c_nativeDim + localCol;

					payload[blockOffset + localOffset] = val;
				}
			}

			FPGAResponse fpgaResponse;
			m_bsclient->SendSync(p_dstIP, fpgaRequest, fpgaResponse);
			assert(fpgaResponse.GetStatus() == FPGA_STATUS_SUCCESS);

			return;
		}

	public:

		LSTMClientImpl(bool enableAsync)
		{
			m_bsclient = BrainSliceClient::Create(enableAsync);
		}

		~LSTMClientImpl()
		{
			delete m_bsclient;
		}

		bool LoadParameters(const uint32_t p_dstIP, std::vector<std::vector<dword_t> > &p_parameters)
		{
			assert(p_parameters.size() == 12);

			struct {
				char name[64];
				size_t matAddr;
				size_t rows;
				size_t cols;
			} weights[] = {
				{ "W_xi", 0, c_outputDim, c_inputDim },
				{ "W_xf", 6, c_outputDim, c_inputDim },
				{ "W_xc", 12, c_outputDim, c_inputDim },
				{ "W_xo", 18, c_outputDim, c_inputDim },
				{ "W_hi", 2, c_outputDim, c_outputDim },
				{ "W_hf", 8, c_outputDim, c_outputDim },
				{ "W_hc", 14, c_outputDim, c_outputDim },
				{ "W_ho", 20, c_outputDim, c_outputDim }
			};

			_LoadBiasVectors(p_dstIP, p_parameters);

			for (int i = 0; i < _countof(weights); ++i)
			{
				_LoadWeightMatrix(p_dstIP, p_parameters[4 + i], weights[i].rows, weights[i].cols, weights[i].matAddr);
			}

			return true;
		}

		bool Evaluate(const uint32_t p_dstIP, const vector<vector <dword_t>> &p_inputVectors, const vector<dword_t> &p_outputVector)
		{
			assert(p_inputVectors.size() > 0);
			assert(p_outputVector.size() == c_outputDim);

			FPGARequest fpgaRequest(0);
			{
				size_t numPhysVectorsPerInput = (c_inputDim + c_nativeDim - 1) / c_nativeDim;
				size_t payloadBytes = p_inputVectors.size() * numPhysVectorsPerInput * c_nativeDim * sizeof(dword_t);
				vector<uint32_t> aux = { (uint32_t)c_EvalFunc, (uint32_t)p_inputVectors.size() };
				char *payload;

				fpgaRequest = m_bsclient->CreateRequest(p_dstIP, aux, (void **)&payload, payloadBytes);

				memset(payload, 0x0, payloadBytes);
				for (int i = 0; i < p_inputVectors.size(); ++i)
				{
					assert(p_inputVectors[i].size() == c_inputDim);
					memcpy(
						payload + i * numPhysVectorsPerInput * c_nativeDim * sizeof(dword_t),
						(char *)&p_inputVectors[i][0],
						c_inputDim * sizeof(dword_t));
				}
			}

			FPGAResponse fpgaResponse;
			m_bsclient->SendSync(p_dstIP, fpgaRequest, fpgaResponse);
			assert(fpgaResponse.GetStatus() == FPGA_STATUS_SUCCESS);
			{
				size_t numPhysVectorsPerOutput = (c_outputDim + c_nativeDim - 1) / c_nativeDim;

				vector<uint32_t> resp_aux;
				size_t responseSize;
				dword_t *payload = (dword_t *)m_bsclient->DecodeResponse(fpgaResponse, resp_aux, responseSize);

				assert((resp_aux.size() == 1) && (resp_aux[0] == numPhysVectorsPerOutput) && (responseSize == (numPhysVectorsPerOutput * c_nativeDim * sizeof(dword_t))));
				memcpy((void *)&p_outputVector[0], payload, c_outputDim * sizeof(dword_t));
			}

			return true;
		}

        bool TestLoopback(const uint32_t p_dstIP, const std::vector<dword_t> &p_inputVector, std::vector<dword_t> &p_outputVector)
        {
            assert(p_inputVector.size() == c_nativeDim);

            p_outputVector.resize(p_inputVector.size());

			FPGARequest fpgaRequest(0);
			{
				size_t payloadBytes = p_inputVector.size() * sizeof(dword_t);
                vector<uint32_t> aux = { (uint32_t)c_LoopbackFunc };
				char *payload;

				fpgaRequest = m_bsclient->CreateRequest(p_dstIP, aux, (void **)&payload, payloadBytes);

                memcpy(payload, &p_inputVector[0], payloadBytes);
			}

			FPGAResponse fpgaResponse;
			m_bsclient->SendSync(p_dstIP, fpgaRequest, fpgaResponse);
			assert(fpgaResponse.GetStatus() == FPGA_STATUS_SUCCESS);
			{
				vector<uint32_t> resp_aux;
				size_t responseSize;
				dword_t *payload = (dword_t *)m_bsclient->DecodeResponse(fpgaResponse, resp_aux, responseSize);

				assert((resp_aux.size() == 1) && (responseSize == (p_inputVector.size() * sizeof(dword_t))));
				memcpy((void *)&p_outputVector[0], payload, p_inputVector.size() * sizeof(dword_t));
			}

            return true;
        }

		LSTMInfo Info(const uint32_t /*p_dstIP*/) // TO-DO: query FPGA at runtime
		{
			LSTMInfo info;
			info.inputDim = c_inputDim;
			info.outputDim = c_outputDim;
			info.nativeDim = c_nativeDim;
			info.modelHash = 0x0;
			return info;
		}
	};

	LSTMClient *LSTMClient::Create(bool asyncSend)
	{
		return new LSTMClientImpl(asyncSend);
	};

}