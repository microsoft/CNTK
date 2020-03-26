#pragma once


// TODO: interpret different kinds properly
// See http://www1.icsi.berkeley.edu/Speech/docs/HTKBook/node67_ct.html
#include <Winsock2.h>

#include "CWaveReader.h"
#include "CFeatExtractor.h"

#pragma comment(lib, "Ws2_32.lib")
class IFeatReader 
{
public:
    virtual ~IFeatReader() {};
    virtual size_t Read(float* buf, size_t cnt) = 0;
};

class WavFileFeatReader : public IFeatReader
{
public:
    UNIMIC_ALIGNED_NEW(WavFileFeatReader);

    WavFileFeatReader(const wchar_t* path)
        : m_reader(path)
    {
        m_sampleCnt = m_reader.SampleCount();
    }

    virtual size_t Read(float* buf, size_t cnt)
    {
        size_t frameCnt = cnt / CFeatExtractor::FeatDim;
        rassert_eq(frameCnt * CFeatExtractor::FeatDim, cnt);

        size_t resultCnt = 0;
        for (size_t i = 0;i < frameCnt; i++)
        {
            if (ReadOneFrame())
            {
                m_extract.Convert(m_inbuf, CFeatExtractor::WindowSize, &buf[i * CFeatExtractor::FeatDim], CFeatExtractor::FeatDim);
                resultCnt += CFeatExtractor::FeatDim;
            }
            else
                break;
        }

        return resultCnt;
    }

private:
    bool ReadOneFrame()
    {
        if (m_ndx >= m_sampleCnt)
            return false;

        size_t sz = (m_ndx == 0 ? (CFeatExtractor::WindowSize) : (CFeatExtractor::Advance));
        size_t cnt = (m_sampleCnt - m_ndx) > sz ? sz : (m_sampleCnt - m_ndx);
        int raw = 0;

        if (m_ndx == 0)
        {
            for (size_t i = 0; i < cnt; i++)
            {
                m_reader.ReadNextSample(&raw);
                m_inbuf[i] = (float)raw;
            }
        }
        else
        {
            memmove(m_inbuf, &m_inbuf[CFeatExtractor::Advance], (CFeatExtractor::WindowSize - CFeatExtractor::Advance) * sizeof(float));
            for (size_t i = 0; i < cnt; i++)
            {
                m_reader.ReadNextSample(&raw);
                m_inbuf[CFeatExtractor::WindowSize - CFeatExtractor::Advance + i] = (float)raw;
            }
        }
        m_ndx += cnt;

        return true;
    }

    CWaveReader m_reader;
    CFeatExtractor m_extract;
    size_t m_ndx = 0;
    size_t m_sampleCnt = 0;
    float m_inbuf[CFeatExtractor::WindowSize];
};

class HTKFeatReader : public IFeatReader
{
public:
    HTKFeatReader(const wchar_t* path)
    {
        rassert_eq(0, _wfopen_s(&m_fp, path, L"rb"));

        int x;
        rassert_eq(1, fread(&x, sizeof(x), 1, m_fp));
        auto nSamples = ntohl(x);
        rassert_op(nSamples, >, 0u);

        rassert_eq(1, fread(&x, sizeof(x), 1, m_fp));
        auto sampPeriod = ntohl(x);
        rassert_op(sampPeriod, >, 0u);

        short y;
        rassert_eq(1, fread(&y, sizeof(y), 1, m_fp));
        auto sampSize = ntohs(y);
        rassert_op(sampSize, >, 0u);

        rassert_eq(1, fread(&y, sizeof(y), 1, m_fp));
        auto paramKind = ntohs(y);
        rassert_eq(paramKind, 9u);

        m_cbTotal = (size_t)nSamples * (size_t)sampSize;

        // printf("%d samples; period %gms; size %hdB; kind %hd\n",
        //         nSamples,
        //         (double)sampPeriod / 10000.0,
        //         sampSize,
        //         paramKind);
    }

    ~HTKFeatReader()
    {
        fclose(m_fp);
    }

    virtual size_t Read(float* buf, size_t cnt)
    {
        auto cb = cnt * sizeof(float);
        auto n = fread(buf, 1, cb, m_fp);
        m_cbRead += n;

        if (n == cb)
        {
            rassert_op(m_cbRead, <=, m_cbTotal);
        }
        else
        {
            rassert_op(n, <, cb);
            rassert_op(feof(m_fp), !=, 0);
            rassert_op(ferror(m_fp), ==, 0);
            rassert_eq(m_cbRead, m_cbTotal);
        }

        rassert_op(n % sizeof(float), ==, 0);
        static_assert(sizeof(float) == sizeof(uint32_t), "ntohl for float");
        auto p = (uint32_t*)buf;
        for (size_t i = 0; i < n / sizeof(float); i++)
            p[i] = ntohl(p[i]);

        return n / sizeof(float);
    }

private:
    FILE* m_fp = nullptr;
    size_t m_cbTotal = 0;
    size_t m_cbRead = 0;
};


class CFeatReader
{
public:
    CFeatReader(const std::wstring& spec, size_t featDim)
        : m_featDim(featDim),
          m_buf(std::make_unique<float[]>(featDim))
    {
        auto n = spec.find_last_of(L'.');
        if (n == std::wstring::npos)
            n = 0;
        auto ext = spec.substr(n);

        if (ext == L".mfc" || ext == L".MFC")
            m_reader = std::make_unique<HTKFeatReader>(spec.c_str());
        else if (ext == L".wav" || ext == L".WAV")
            m_reader = std::make_unique<WavFileFeatReader>(spec.c_str());
        else
            rfail("unknown extension");
    }

    const float* Feats() const
    {
        return m_buf.get();
    }

    bool Forward(size_t stride)
    {
        size_t offset;
        size_t cnt;

        if (m_start)
        {
            offset = 0;
            cnt = m_featDim;
            m_start = false;
        }
        else
        {
            rassert_op(stride, <=, m_featDim);
            memcpy(&m_buf[0], &m_buf[stride], (m_featDim - stride) * sizeof(m_buf[0]));

            offset = m_featDim - stride;
            cnt = stride;
        }

        auto n = m_reader->Read(&m_buf[offset], cnt);
        if (n != cnt)
        {
            rassert_op(n, <, cnt);
            m_endPadding += (cnt - n);
            // CNTK RowSlice does not do this
            // memset(&m_buf[offset + n], 0, (cnt - n) * sizeof(m_buf[0]));
        }

        return m_endPadding < m_featDim;
    }

private:
    std::unique_ptr<IFeatReader> m_reader;
    size_t m_featDim;
    bool m_start = true;
    size_t m_endPadding = 0;
    std::unique_ptr<float[]> m_buf;
};
