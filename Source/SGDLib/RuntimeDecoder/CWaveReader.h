#pragma once

struct WaveFormat
{
    uint16_t AudioFormat;
    uint16_t NumChannels;
    uint32_t SampleRate;
    uint32_t ByteRate;
    uint16_t BlockAlign;
    uint16_t BitsPerSample;
};

class CWaveReader
{
public:
    CWaveReader(const TCHAR* path)
    {
        _ftprintf(stderr, _T("(R) %s\n"), path);
        rassert_eq1(0, _tfopen_s(&fp, path, _T("rb")));
        fmt = DetectWaveFormat(fp);

        Rewind();
    }

    ~CWaveReader()
    {
        fclose(fp);
        fp = nullptr;
    }

    const WaveFormat& Format() const { return fmt; }
    uint32_t DataSize() const { return dataSize; }
    size_t SampleCount() const
    {
        auto u64SampleCnt = (uint64_t)DataSize() * 8 / (uint64_t)Format().BitsPerSample;
        auto sampleCnt = (size_t)u64SampleCnt;
        rassert_eq1((uint64_t)sampleCnt, u64SampleCnt);
        return sampleCnt;
    }

    void Skip(uint64_t begin)
    {
        auto begin_bit = begin * (uint64_t)fmt.BitsPerSample;
        auto begin_byte = begin_bit / 8;
        rassert_eq1(0, _fseeki64(fp, begin_byte, SEEK_CUR));

        byte = begin_byte;
        c = EOF;
        i = begin;
    }

    void Rewind()
    {
        rassert_eq1(0, fseek(fp, 0, SEEK_SET));
        dataSize = SkipWaveHeader(fp);

        Skip(0);
    }

    float LoopReadNextSample(int* pRaw = nullptr)
    {
        if (i >= SampleCount())
            Rewind();

        return ReadNextSample(pRaw);
    }

    float ReadNextSample(int* pRaw = nullptr)
    {
        auto bit = i * (uint64_t)fmt.BitsPerSample;
        auto _max = (1 << fmt.BitsPerSample) - 1;

        float result;
        if (fmt.BitsPerSample <= 8U)
        {
            if (c == EOF)
            {
                rassert_op(byte, <, dataSize);
                c = fgetc(fp);
                rassert_op(c, !=, EOF);
            }

            rassert_op(bit % 8 + fmt.BitsPerSample, <=, 8U);

            auto raw = (c >> (bit % 8)) & _max;
            result = 2 * (float)raw / (float)_max - 1.0f;

            if (bit % 8 + fmt.BitsPerSample == 8)
            {
                c = EOF;
                byte++;
            }

            if (pRaw)
                *pRaw = raw;
        }
        else
        {
            rassert_eq1(0U, bit % 8U);
            rassert_eq1(0U, fmt.BitsPerSample % 8U);
            rassert_op(fmt.BitsPerSample, <=, 32);
            int raw = 0;
            for (int b = 0; b < fmt.BitsPerSample; b += 8)
            {
                rassert_op(byte, <, dataSize);
                c = _fgetc_nolock(fp);
                rassert_op(c, !=, EOF);
                byte++;
                raw |= ((c & 0xff) << b);
            }
            raw = (raw << (32 - fmt.BitsPerSample)) >> (32 - fmt.BitsPerSample);
            result = (float)raw / (1ull << (fmt.BitsPerSample - 1));

            if (pRaw)
                *pRaw = raw;
        }

        i++;

        return result;
    }

private:
    WaveFormat DetectWaveFormat(FILE* fp1)
    {
        char buf[4];
        rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
        rassert_eq1(buf[0], 'R');
        rassert_eq1(buf[1], 'I');
        rassert_eq1(buf[2], 'F');
        rassert_eq1(buf[3], 'F');

        uint32_t chunkSize;
        rassert_eq1(4U, fread_s(&chunkSize, sizeof(chunkSize), 1, sizeof(chunkSize), fp1));

        // WAVE chunk
        rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
        rassert_eq1(buf[0], 'W');
        rassert_eq1(buf[1], 'A');
        rassert_eq1(buf[2], 'V');
        rassert_eq1(buf[3], 'E');

        // find fmt chunk
        bool found = false;
        while (!found)
        {
            rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
            if (buf[0] == 'f' &&
                buf[1] == 'm' &&
                buf[2] == 't' &&
                buf[3] == ' ')
                found = true;

            rassert_eq1(4U, fread_s(&chunkSize, sizeof(chunkSize), 1, sizeof(chunkSize), fp1));

            if (!found) // skip this chunk
            {
                auto temp = std::make_unique<char[]>(chunkSize);
                rassert_eq1(chunkSize, fread_s(temp.get(), chunkSize, 1, chunkSize, fp1));
            }
        }
        rassert_eq1(found, true);

        // now read format information
        WaveFormat r;
        rassert_eq1(sizeof(r), fread_s(&r, sizeof(r), 1, sizeof(r), fp1));
        return r;
    }

    uint32_t SkipWaveHeader(FILE* fp1)
    {
        char buf[4];
        rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
        rassert_eq1(buf[0], 'R');
        rassert_eq1(buf[1], 'I');
        rassert_eq1(buf[2], 'F');
        rassert_eq1(buf[3], 'F');

        uint32_t chunkSize;
        rassert_eq1(4U, fread_s(&chunkSize, sizeof(chunkSize), 1, sizeof(chunkSize), fp1));

        // WAVE chunk
        rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
        rassert_eq1(buf[0], 'W');
        rassert_eq1(buf[1], 'A');
        rassert_eq1(buf[2], 'V');
        rassert_eq1(buf[3], 'E');

        // try to find data chunk
        bool found = false;
        while (!found)
        {
            rassert_eq1(4U, fread_s(buf, sizeof(buf), 1, 4, fp1));
            if (buf[0] == 'd' &&
                buf[1] == 'a' &&
                buf[2] == 't' &&
                buf[3] == 'a')
                found = true;

            rassert_eq1(4U, fread_s(&chunkSize, sizeof(chunkSize), 1, sizeof(chunkSize), fp1));

            if (!found) // skip this chunk
            {
                auto temp = std::make_unique<char[]>(chunkSize);
                rassert_eq1(chunkSize, fread_s(temp.get(), chunkSize, 1, chunkSize, fp1));
            }
        }
        rassert_eq1(found, true);

        return chunkSize;
    }

private:
    FILE* fp;
    WaveFormat fmt;
    uint32_t dataSize;

    uint64_t i;
    int c;
    uint64_t byte;
};
