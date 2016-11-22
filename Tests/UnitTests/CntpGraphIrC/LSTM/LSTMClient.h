#pragma once

#include <stdint.h>
#include <vector>

namespace LSTM
{
	typedef short dword_t;

	class LSTMClient
	{
	public:

		typedef struct {
			uint32_t modelHash;
			uint32_t inputDim;
			uint32_t outputDim;
			uint32_t nativeDim;
		} LSTMInfo;

		// Create LSTMClient instance.
		static LSTMClient *Create(bool asyncSend = false);

		// Load LSTM parameters.
		//     p_dstIP = IP address of destination FPGA.
		//     p_parameters = vector of matrices and biases, which are expected in this order:
		//         b_i, b_f, b_c, b_o, W_xi, W_xf, W_xc, W_xo, W_hi, W_hf, W_hc, W_ho 
        //
		//     bias vector length = 'outputDim'
		//     W_x* matrices are of dimension 'outputDim' x 'inputDim'
		//     W_h* matrices are of dimension 'outputDim' x 'outputDim'
		// 
        //     All matrices are stored in row-order, i.e., 
        //         element of matrix 'm' at matrix coordinate (r, c) is stored at p_parameters[m][r * dim + c]
        //
		virtual bool LoadParameters(const uint32_t p_dstIP, std::vector<std::vector<dword_t> > &p_parameters) = 0;

		// Send 1 or more input vectors to BrainSlice FPGA for processing. This call is synchronous and will block
        // until a response is returned from the FPGA.
		//	   p_dstIP = IP address of destination FPGA.
		//	   p_inputVectors = vector of input vectors, each of dimension 'inputDim'
		//     p_outputVector = storage for output vector, of dimension 'outputDim'
		virtual bool Evaluate(const uint32_t p_dstIP, const std::vector<std::vector<dword_t> > &p_inputVectors, const std::vector<dword_t> &p_outputVector) = 0;

        virtual bool TestLoopback(const uint32_t p_dstIP, const std::vector<dword_t> &p_inputVector, std::vector<dword_t> &p_outputVector) = 0;

		// Query BrainSlice capabilities
		virtual LSTMInfo Info(const uint32_t p_dstIP) = 0;
	};
}