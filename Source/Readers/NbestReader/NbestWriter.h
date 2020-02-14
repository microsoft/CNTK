#pragma once
#include "DataWriter.h"
#include "ScriptableObjects.h"
#include <map>
#include <vector>

#define NBEST_SAVE_TXT 0
#define NBEST_SAVE_HTKLATTICE 1

namespace Microsoft
{
	namespace MSR
	{
		namespace CNTK
		{

			template <class ElemType>
			class NbestWriter : public IDataWriter
			{
			private:
				std::vector<std::wstring> outputFiles;
				size_t outputFileIndex;
                void SaveTxt(std::wstring& outputFile, std::vector<std::pair<std::vector<size_t>, ElemType>>& outputData);
				int m_verbosity;
				size_t m_overflowWarningCount;
				size_t m_maxNumOverflowWarning;
				float m_overflowValue;
                size_t saveType;

				enum OutputTypes
				{
					outputReal,
					outputCategory,
				};

			public:
				template <class ConfigRecordType>
				void InitFromConfig(const ConfigRecordType& writerConfig);
				virtual void Init(const ConfigParameters& config)
				{
					InitFromConfig(config);
				}
				virtual void Init(const ScriptableObjects::IConfigRecord& config)
				{
					InitFromConfig(config);
				}
				virtual void Destroy();
				virtual void GetSections(std::map<std::wstring, SectionType, nocase_compare>& sections);
				virtual bool SaveData(size_t recordStart, const std::map<std::wstring, void*, nocase_compare>& matrices, size_t numRecords, size_t datasetSize, size_t byteVariableSized);
				virtual void SaveMapping(std::wstring saveId, const std::map<LabelIdType, LabelType>& labelMapping);
				virtual bool SupportMultiUtterances() const
				{
					return false;
				};
			};
		} // namespace CNTK
	} // namespace MSR
} // namespace Microsoft
