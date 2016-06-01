#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>


namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			class DiskCache {
			public:
				DiskCache(size_t maxCapacity);
				bool Write(void* buffer, size_t size);
				void ResetReadPos();
				size_t Read(void* buffer);
				size_t CachedBlocksNum();

			private:
				void OpenCacheFile();
				void InitCacheDir(const char* dirPath, int maxFileIndex);

			private:
				std::fstream m_cacheFile;
				std::vector<std::pair<size_t, size_t> > m_blocks;  //pos, size
				size_t m_cachedSize;
				size_t m_maxCapacity;
				size_t m_readIndex;
			};


			class MemCache {
			public:
				MemCache(size_t maxCapacity);
				bool Write(void* buffer, size_t size);
				void ResetReadPos();
				size_t Read(void* buffer);
				size_t CachedBlocksNum();
				size_t Capacity();
				~MemCache();

			private:
				void AllocMem(int maxCapacity);

			private:
				std::vector<std::pair<void*, size_t> > m_blocks;
				size_t m_cachedSize;
				size_t m_maxCapacity;
				size_t m_readIndex;
				char* m_memBlock;
			};
		}
	}
}