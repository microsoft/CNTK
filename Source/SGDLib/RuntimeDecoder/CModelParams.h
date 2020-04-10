#pragma once
struct NODEINFO
{
    std::wstring name;
    size_t M;
    size_t N;
    float* data;
};

//
// Binary model parameter file support
//
static const char* MODEL_SIG = "[Model Parameters Binary File v0.01]";
class CModelParams
{
public:
    
    CModelParams(
            const wchar_t* path,
            MatrixKind matrixKind,
            const std::set<std::wstring>& transposeMatrices,
            const std::set<std::wstring>& floatMatrices)
    {
        FILE* fp = nullptr;
        rassert_eq(0, _wfopen_s(&fp, path, L"rb"));

        char sig[64];
        rassert_eq(1, fread(sig, strlen(MODEL_SIG), 1, fp));
        rassert_eq(0, strncmp(sig, MODEL_SIG, strlen(MODEL_SIG)));

        std::vector<std::future<bool>> fread_futures;

        for (;;)
        {
            uint32_t nameLen;
            auto n = fread(&nameLen, sizeof(nameLen), 1, fp);
            if (n != 1)
            {
                rassert_op(n, <, 1u);
                rassert_op(feof(fp), !=, 0);
                rassert_op(ferror(fp), ==, 0);
                break;
            }

            auto name = std::make_unique<wchar_t[]>(nameLen + 1);
            rassert_eq(1, fread(&name[0], nameLen * sizeof(name[0]), 1, fp));
            name[nameLen] = L'\0';

            uint32_t Dims[2];
            rassert_eq(1, fread(Dims, sizeof(Dims), 1, fp));
            auto M = Dims[0];
            auto N = Dims[1];

            if (N == 1)
            {
                auto p = m_vectorMap.emplace(name.get(), CVector(M));
                rassert_eq(p.second, true);
                rassert_eq(M, fread(&p.first->second[0], sizeof(float), M, fp));
            }
            else
            {
                bool transpose =
                        transposeMatrices.find(name.get()) != transposeMatrices.end();
                bool flt =
                        floatMatrices.find(name.get()) != floatMatrices.end();
                auto p = m_matrixMap.emplace(name.get(),
                        transpose ?
                            make_unique_matrix(N, M, flt ? MatrixKind::Float : matrixKind) :
                            make_unique_matrix(M, N, flt ? MatrixKind::Float : matrixKind));
                rassert_eq(p.second, true);
                fread_futures.push_back(p.first->second->fread(fp, transpose));
            }
        }

        fclose(fp);

        for (auto& ff: fread_futures)
            ff.get();

        // for (const auto& p: m_paramMap)
        // {
        //     printf("%S=LearnableParameter [%u,%u]\n", p.first.c_str(), p.second.M, p.second.N);
        //     for (size_t i = 0; i < p.second.M; i++)
        //     {
        //         for (size_t j = 0; j < p.second.N; j++)
        //             printf(" %.9g", p.second.X(i, j));
        //         printf(" \n");
        //     }
        //     printf(" #################################################################### \n");
        // }
    }
    
    CModelParams(std::vector<NODEINFO>& vt_nodes,
        MatrixKind matrixKind,
        const std::set<std::wstring>& transposeMatrices,
        const std::set<std::wstring>& floatMatrices)
    {
        
        std::vector<std::future<bool>> fread_futures;

        
        // int debugcount = 0;
        for (auto& nd : vt_nodes)
        {
            // fprintf(stderr, "count = %d\n", debugcount);
            auto M = nd.M;
            auto N = nd.N;

            if (N == 1)
            {
                auto p = m_vectorMap.emplace(nd.name, CVector(M));
                rassert_eq(p.second, true);
                memcpy(&p.first->second[0], nd.data, sizeof(float)* M);
            }
            else
            {
                bool transpose =
                    transposeMatrices.find(nd.name) != transposeMatrices.end();
                bool flt =
                    floatMatrices.find(nd.name) != floatMatrices.end();
                auto p = m_matrixMap.emplace(nd.name,
                                             transpose ? make_unique_matrix(N, M, flt ? MatrixKind::Float : matrixKind) : make_unique_matrix(M, N, flt ? MatrixKind::Float : matrixKind));
                rassert_eq(p.second, true);
                fread_futures.push_back(p.first->second->mread(nd.data, transpose));
            }
            
            //debugcount++;

        }

        for (auto& ff : fread_futures)
            ff.get();

    }
    
    static void Compile(const wchar_t* inPath, const wchar_t* outPath)
    {

        FILE* infp = nullptr;
        rassert_eq(0, _wfopen_s(&infp, inPath, L"r"));

        FILE* outfp = nullptr;
        rassert_eq(0, _wfopen_s(&outfp, outPath, L"wb"));

        std::wstring line;
        rassert_eq(true, _getline(infp, line));
        if (line != L" ")
            rfail("first line not empty");

        const wchar_t* LP = L"LearnableParameter ";
        const size_t LPLen = wcslen(LP);
        const wchar_t* POUNDS = L" #################################################################### ";
        const wchar_t* POUNDS_SHORT = L" ####################################################################";

        rassert_eq(1, fwrite(MODEL_SIG, strlen(MODEL_SIG), 1, outfp));
        while (_getline(infp, line))
        {
            const wchar_t *p;
            for (p = line.c_str(); *p != L'\0'; p++)
                if (*p == L'=')
                    break;
            rassert_eq(*p++, L'=');

            if (wcsncmp(p, LP, LPLen) == 0)
            {
                auto nameLen = (uint32_t)(p - line.c_str() - 1);
                rassert_eq(1, fwrite(&nameLen, sizeof(nameLen), 1, outfp));
                rassert_eq(1, fwrite(line.c_str(), nameLen * sizeof(line[0]), 1, outfp));

                p += LPLen;
                rassert_eq(*p++, L'[');
                uint32_t Dims[2];
                #ifdef LINUXRUNTIMECODE
                rassert_eq(2, swscanf(p, L"%u,%u]", &Dims[0], &Dims[1]));
                #else
                rassert_eq(2, swscanf_s(p, L"%u,%u]", &Dims[0], &Dims[1]));
                #endif
                rassert_eq(1, fwrite(Dims, sizeof(Dims), 1, outfp));

                auto M = Dims[0];
                auto N = Dims[1];
                auto x = std::make_unique<float[]>(M * N);
                for (uint32_t i = 0; i < M; i++)
                {
                    for (size_t j = 0; j < N; j++)
                        rassert_eq(1, fscanf_s(infp, " %f", &x[i * N + j]));

                    rassert_eq(true, _getline(infp, line));
                    rassert_eq(0, wcscmp(line.c_str(), L" "));
                }
                rassert_eq(M * N, fwrite(&x[0], sizeof(x[0]), M * N, outfp));

                if (_getline(infp, line))
                    rassert_eq(0, wcscmp(line.c_str(), POUNDS));
                else
                    rassert_eq(0, wcscmp(line.c_str(), POUNDS_SHORT));
            }
        }

        fclose(outfp);
        fclose(infp);
    }

    const IMatrix* GetMatrixParams(const std::wstring& name) const
    {
        auto iter = m_matrixMap.find(name);
        if (iter == m_matrixMap.end())
            rfail("params not found: %S", name.c_str());
        return iter->second.get();
    }

    void SetMatrixParams(const std::wstring& name, std::unique_ptr<IMatrix>&& A)
    {
        auto iter = m_matrixMap.find(name);
        if (iter == m_matrixMap.end())
            rfail("params not found: %S", name.c_str());
        iter->second = std::move(A);
    }

    void RemoveMatrixParams(const std::wstring& name)
    {
        auto iter = m_matrixMap.find(name);
        if (iter == m_matrixMap.end())
            rfail("params not found: %S", name.c_str());
        m_matrixMap.erase(iter);
    }

    const CVector& GetVectorParams(const std::wstring& name) const
    {
        auto iter = m_vectorMap.find(name);
        if (iter == m_vectorMap.end())
            rfail("params not found: %S", name.c_str());
        return iter->second;
    }

    CVector& AddVectorParams(const std::wstring& name, uint32_t M)
    {
        auto p = m_vectorMap.emplace(name, M);
        rassert_eq(p.second, true);
        return p.first->second;
    }

    const std::set<std::wstring> GetParamNames() const
    {
        std::set<std::wstring> result;
        for (const auto& entry: m_matrixMap)
            rassert_eq(result.insert(entry.first).second, true);
        for (const auto& entry: m_vectorMap)
            rassert_eq(result.insert(entry.first).second, true);
        return result;
    }

    void RenameParam(
            const std::wstring& prevName,
            const std::wstring& newName)
    {
        auto iter1 = m_matrixMap.find(prevName);
        if (iter1 != m_matrixMap.end())
        {
            auto v = std::move(iter1->second);
            m_matrixMap.erase(iter1);
            auto p = m_matrixMap.emplace(newName, std::move(v));
            rassert_eq(p.second, true);
            return;
        }

        auto iter2 = m_vectorMap.find(prevName);
        if (iter2 != m_vectorMap.end())
        {
            auto v = std::move(iter2->second);
            m_vectorMap.erase(iter2);
            auto p = m_vectorMap.emplace(newName, std::move(v));
            rassert_eq(p.second, true);
            return;
        }

        rfail("parameter not found: %S", prevName.c_str());
    }

    void Save(const wchar_t* outPath)
    {
        FILE* outfp = nullptr;
        rassert_eq(0, _wfopen_s(&outfp, outPath, L"wb"));

        rassert_eq(1, fwrite(MODEL_SIG, strlen(MODEL_SIG), 1, outfp));

        for (const auto& p: m_matrixMap)
        {
            const auto& name = p.first;
            auto nameLen = (uint32_t)name.size();
            rassert_eq(1, fwrite(&nameLen, sizeof(nameLen), 1, outfp));
            rassert_eq(1, fwrite(name.c_str(), nameLen * sizeof(name[0]), 1, outfp));

            auto A = p.second.get();
            A->fwrite(outfp);
        }

        for (const auto& p: m_vectorMap)
        {
            const auto& name = p.first;
            auto nameLen = (uint32_t)name.size();
            rassert_eq(1, fwrite(&nameLen, sizeof(nameLen), 1, outfp));
            rassert_eq(1, fwrite(name.c_str(), nameLen * sizeof(name[0]), 1, outfp));

            const auto& A = p.second;
            uint32_t Dims[2] = { A.M, 1 };
            rassert_eq(1, fwrite(Dims, sizeof(Dims), 1, outfp));
            rassert_eq(A.M, fwrite(&A[0], sizeof(A[0]), A.M, outfp));
        }

        fclose(outfp);
    }

private:
    std::map<std::wstring, std::unique_ptr<IMatrix>> m_matrixMap;
    std::map<std::wstring, CVector> m_vectorMap;
};
