#include "Cache.h"
#include "CDenseReader.h"

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif
#include <iostream>

using namespace std;



namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			void DiskCache::InitCacheDir(const char* dirPath, int maxFileIndex) {
				//std C++ has no protable code to iterate a dir
#ifdef _WIN32
				int flag = _mkdir(dirPath);
#else
				int flag = mkdir(dirPath, 0755);
#endif
				//clear outdated cache files
				if (flag != 0) {
					char buf[255];
					for (int i = 0; i < maxFileIndex; ++i) {
						sprintf(buf, "%s/%d", dirPath, i);
						remove(buf);
					}
				}
			}

			void DiskCache::OpenCacheFile() {
				//cache dir path
#ifdef _WIN32
				const char dirName[] = "./cache";
#else
				const char dirName[] = "/tmp/cachefromcdensereader";
#endif
				const int maxFileIndex = 100;

				//create and clean cache dir
				InitCacheDir(dirName, maxFileIndex);

				//try open a cache file
				char buf[255];
				for (int i = 0; i < maxFileIndex; ++i) {
					sprintf(buf, "%s/%d", dirName, i);
					this->m_cacheFile.open(buf, ios::in | ios::out | ios::binary | ios::trunc);
					if (this->m_cacheFile) {
						break;
					}
				}
				if (!this->m_cacheFile) {
					RuntimeError("allocate cache file failed");
				}
			}



			DiskCache::DiskCache(size_t maxCapacity) {
				this->OpenCacheFile();
				this->m_cachedSize = 0;
				this->m_maxCapacity = maxCapacity;
				this->ResetReadPos();
			}

			bool DiskCache::Write(void* buffer, size_t size) {
				if (this->m_cachedSize + size > this->m_maxCapacity) {
					return false;
				}
				this->m_cachedSize += size;
				size_t pos = this->m_cacheFile.tellg();
				this->m_blocks.push_back({ pos, size });
				this->m_cacheFile.write((char*)buffer, size);
				return true;
			}

			void DiskCache::ResetReadPos() {
				this->m_readIndex = 0;
			}

			size_t DiskCache::Read(void* buffer) {
				if (this->m_readIndex >= this->CachedBlocksNum()) {
					RuntimeError("read disk cache out of range");
				}
				auto& block = this->m_blocks[this->m_readIndex++];
				this->m_cacheFile.seekg(block.first, ios::beg);
				this->m_cacheFile.read((char*)buffer, block.second);
				return block.second;
			}

			size_t DiskCache::CachedBlocksNum() {
				return this->m_blocks.size();
			}


			MemCache::MemCache(size_t maxCapacity) {
				this->ResetReadPos();
				this->AllocMem(maxCapacity);
				this->m_cachedSize = 0;
			}

			void MemCache::AllocMem(size_t maxCapacity) {
				while (maxCapacity > 0) {
					this->m_memBlock = (char*)malloc(maxCapacity);
					if (this->m_memBlock != NULL) {
						this->m_maxCapacity = maxCapacity;
						break;
					}
					maxCapacity = (size_t)(maxCapacity * 0.8);
				}
			}

			size_t MemCache::Capacity() {
				return this->m_maxCapacity;
			}

			MemCache::~MemCache() {
				delete[] this->m_memBlock;
			}

			bool MemCache::Write(void* buffer, size_t size) {
				if (size + this->m_cachedSize > this->m_maxCapacity) {
					return false;
				}
				void* memBlock = m_memBlock + m_cachedSize;
				memcpy(memBlock, buffer, size);
				this->m_blocks.push_back({ memBlock, size });
				this->m_cachedSize += size;
				return true;
			}

			void MemCache::ResetReadPos() {
				this->m_readIndex = 0;
			}

			size_t MemCache::Read(void* buffer) {
				if (this->m_readIndex >= this->CachedBlocksNum()) {
					RuntimeError("read mem cache out of range");
				}
				auto& block = this->m_blocks[this->m_readIndex++];
				memcpy(buffer, block.first, block.second);
				return block.second;
			}

			size_t MemCache::CachedBlocksNum() {
				return this->m_blocks.size();
			}
		}
	}
}