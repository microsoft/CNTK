//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

inline std::vector<float> get_features(uint8_t* image_data_array, uint32_t reqWidth, uint32_t reqHeight)
{
    uint32_t size = reqWidth * reqHeight * 3;

    // BGR conversion to BBB..GGG..RRR
    std::vector<float> featuresLocal;

    // convert BGR array to BBB...GGG...RRR array
    for (uint32_t c = 0; c < 3; c++) {
        for (uint32_t p = c; p < size; p = p + 3)
        {
            float v = image_data_array[p];
            featuresLocal.push_back(v);
        }
    }
    return featuresLocal;
}

inline int64_t find_class(std::vector<float> outputs)
{
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

inline bool does_file_exist(std::wstring  file_name)
{
    return std::experimental::filesystem::exists(file_name);
}

inline std::vector<std::wstring> read_class_names(const std::wstring filename)
{
    std::vector<std::wstring> classNames;
    std::wifstream fp(filename);
    if (!fp.is_open())
    {
        return classNames;
    }
    std::wstring name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name.substr(name.find(' ') + 1));
    }
    fp.close();
    return classNames;
}