//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibraryInternals.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#pragma warning(push)
#pragma warning(disable : 4244 4245)
#include <boost/crc.hpp>
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "tensorboard/tensorboard.pb.h"
#pragma warning(pop)

#include "BackCompat.h"
#include "fileutil.h"
#include "hostname.h"
#include "tensorboard/TensorBoardUtils.h"
#include "Utils.h"

namespace CNTK
{
    namespace Internal
    {
        template<typename T>
        static void Encode(char* buf, T value)
        {
            // Assume little-endian target encoding.
            memcpy(buf, &value, sizeof(value));
        }

        static uint32_t GetMaskedCrc(const char* data, size_t n)
        {
            // Compute CRC32C.
            boost::crc_optimal<32, 0x1EDC6F41, 0xFFFFFFFF, 0xFFFFFFFF, true, true> crc;
            crc.process_bytes(reinterpret_cast<const void*>(data), n);
            uint32_t value = crc();

            // Rotate right by 15 bits and add a constant.
            return ((value >> 15) | (value << 17)) + 0xa282ead8ul;
        }

        static std::wstring GetNewFilePath(const std::wstring& dir, time_t time)
        {
            std::wostringstream filename;
            if (!dir.empty())
            {
                // Windows can tolerate forward slashes in paths.
                filename << dir << L"/";
            }

            filename << L"events.out.tfevents." 
                << std::setfill(L'0') << std::setw(10) << time
                << L"." << ToWString(GetHostName());
            return filename.str();
        }

        static std::string Serialize(const tensorflow::Event& event)
        {
            std::string record;
            event.AppendToString(&record);
            // Rely on RVO to make returning a string cheap.
            return record;
        }

        TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir, const FunctionPtr& modelToVisualize)
            : m_model(modelToVisualize),
            m_dir(dir),
            m_file(NULL),
            m_fileName()
        {
        }

        TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir,
                                                     const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize)
            : TensorBoardFileWriter(dir, ConvertFromLegacyModel(modelToVisualize))
        {
        }

        void TensorBoardFileWriter::Init()
        {
            time_t time = std::time(0);
            std::wstring filePath = GetNewFilePath(m_dir, time);

            msra::files::make_intermediate_dirs(filePath);

            m_file = fopenOrDie(ToString(filePath), "wb");
            m_fileName = filePath;

            // Write the first record with the current version, and flush
            // right away so the file contents will be easily determined.
            WriteVersion(time);

            if (m_model)
            {
                WriteModel();
            }

            Flush();
        }

        void TensorBoardFileWriter::WriteValue(const std::wstring& name, float value, uint64_t step)
        {
            tensorflow::Event event;
            event.set_step(step);
            event.set_wall_time(static_cast<double>(std::time(0)));

            tensorflow::Summary* summary = event.mutable_summary();
            tensorflow::Summary::Value* summaryValue = summary->add_value();
            summaryValue->set_tag(ToString(name));
            summaryValue->set_simple_value(value);

            WriteRecord(Serialize(event));
        }

        void TensorBoardFileWriter::WriteModel()
        {
            assert(m_model != nullptr);

            // Convert the model to tensorflow GraphDef first.
            tensorflow::GraphDef graph;
            CreateTensorBoardGraph(m_model->RootFunction(), graph);

            std::string graphStr;
            graph.AppendToString(&graphStr);

            // Wrap it as an event.
            tensorflow::Event event;
            event.set_wall_time(static_cast<double>(std::time(0)));
            event.set_graph_def(graphStr);

            WriteRecord(Serialize(event));
        }

        void TensorBoardFileWriter::WriteRecord(const std::string& data)
        {
            if (m_file == NULL)
            {
                Init();
            }

            // Header: record length (uint64_t) + masked CRC of that (uint32_t).
            char header[sizeof(uint64_t) + sizeof(uint32_t)];
            Encode(header, static_cast<uint64_t>(data.size()));
            Encode(header + sizeof(uint64_t), GetMaskedCrc(header, sizeof(uint64_t)));

            // Footer: marked CRC of the actual record.
            char footer[sizeof(uint32_t)];
            Encode(footer, GetMaskedCrc(data.data(), data.size()));

            try
            {
                // Record = Header + Data + Footer.
                fwriteOrDie(header, sizeof(header[0]), sizeof(header), m_file);
                fwriteOrDie(data.data(), sizeof(data[0]), data.size(), m_file);
                fwriteOrDie(footer, sizeof(footer[0]), sizeof(footer), m_file);
            }
            catch (const std::runtime_error&)
            {
                // Close the existing file.
                // If the exception was caught upstream, a new file will be created on subsequent writes.
                fprintf(stderr,
                    "TensorBoardFileWriter: Unable to write to the currently open file. "
                    "Subsequent writes will attempt to re-open a new one. (%ls)", m_fileName.c_str());
                Close();
                throw;
            }
        }

        void TensorBoardFileWriter::WriteVersion(time_t time)
        {
            // Version string present in the first entry of every event file.
            tensorflow::Event event;
            event.set_wall_time(static_cast<double>(time));
            event.set_file_version("brain.Event:2");

            WriteRecord(Serialize(event));
        }

        bool TensorBoardFileWriter::Flush()
        {
            if (m_file == NULL)
            {
                return false;
            }

            if (fflush(m_file))
            {
                fprintf(stderr, "TensorBoardFileWriter: Error flushing the event file (%ls).", m_fileName.c_str());
                return false;
            }

            return true;
        }

        bool TensorBoardFileWriter::Close()
        {
            if (m_file == NULL)
            {
                return false;
            }

            bool success = Flush();
            if (fclose(m_file))
            {
                fprintf(stderr,
                        "TensorBoardFileWriter: Error closing the previous event file (%ls).", m_fileName.c_str());
                success = false;
            }

            m_file = NULL;
            m_fileName.clear();
            return success;
        }
    }
}
