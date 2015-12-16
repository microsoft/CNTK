// History:
//   feiyan, 2014-07-29, creating	
//	 qiwei, 2014-09-23, adding MPI related information, and comments
//
// Summary:
// A multipart message in ZeroMQ (from ROUTER socket) contains a few message
// pieces. The first several ones are address (end with empty message) and the
// following ones are true messages.
// The MessagePackage class contains and deals with multipart message.
// For the design convinent, this MessagePackage should also being message 
// Between MPI intefaces.

// MessagePackage has 3 parts:
//       1. the address that message shuold deliver to. // ? should be from
//       2. empty message for zmq issue
//       3. data which contains head and body.
//              heads are compose of sevral(exactly: 8) integers:
//                    [requestType][send_id][recv_id][send_rank][recv_rank][table_id][reserved][reserved]
//
// 0: RequestType
// 1: SrcId
// 2: DstId
// 3: SrcRank
// 4: DstRank
// 5: CacheId
// 6: TableId
// 7: Count

#ifndef _MULTIVERSO_MSG_PACK_H_
#define _MULTIVERSO_MSG_PACK_H_

#include <vector>
#include "zmq.hpp"
#include "conf.h"

namespace multiverso
{
	class MsgPack
	{
	public:
		MsgPack() : start_(0) {}
		// Parsing a memory block (mainly from MPI) as ZMQ messages.
		MsgPack(char *buffer, int size);
		MsgPack(zmq::socket_t *socket);
		~MsgPack();

		void Clear();
		void Push(zmq::message_t *msg);
		void Send(zmq::socket_t *socket);
		void Serialize(void *buffer, long long *size);
		void GetSerializeSize(long long *size);

		RequestType GetRequestType();
		int GetSrcId();
		int GetSrcRank();
		int GetDstId();
		int GetDstRank();
		int GetCacheId();
		int GetTableId();
		int GetCount();
		int GetMsgCount();
		zmq::message_t *GetMsg(int idx);

		void PushRow(long long row_id, long long row_size, void *row);
		void PushRow(long long row_size, void *row);

		void AsRegister(int src_id, int src_rank);
		MsgPack *GetReplyReg(int response_rank);

		void AsReduce(int src_id, int cache_id);

		void AsGet(int src_id, int src_rank, int dst_rank, int table_id,
			long long *rows, long long num);
		MsgPack *GetReplyGet(int row_id = -1);

		//void AsData(int src_id, int src_rank);
		//MsgPack *GetReplyData();

		void AsAdd(int src_rank, int dst_rank, int table_id);
		void AsAdd(int src_rank, int dst_rank, int table_id, int row_id);
		void AsExit(int src_rank);
		void AsFinish(int src_rank);
		void AsClock(int src_id, int src_rank);
		void AsReplyClock();
		void AsBarrier(int src_id, int src_rank);
		void AsReplyBarrier();

		void AsUpdate(int src_rank, int dst_rank, int table_id, int row_id);

	protected:
		int *GetHeader();
		void ClearWithEmptyAddr();
		MsgPack *CreateMsgPackWithAddress();

		int start_;
		vector<zmq::message_t*> messages_;
	};
}

#endif //_MULTIVERSO_MSG_PACK_H_ 