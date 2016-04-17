#ifndef MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_
#define MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_

#include <multiverso/blob.h>
#include <vector>
#include <cmath>

namespace multiverso {
class QuantizationFilter {
 public:
    QuantizationFilter() {}

    virtual ~QuantizationFilter() {}

    virtual void FilterIn(const std::vector<Blob>& blobs,
        std::vector<Blob>* outputs) = 0;

    virtual void FilterOut(const std::vector<Blob>& blobs,
        std::vector<Blob>* outputs) = 0;
 private:
};

template<typename data_type, typename index_type>
class SparseFilter : public QuantizationFilter {
 public:
    explicit SparseFilter(double clip) :clip_value(clip) {}

    ~SparseFilter() {}

    // Returns compressed blobs given input blobs.
    // Each input blob in vector will generate two blobs in result:
    //  the first blob contains info: compressed or not, original
    //  blob size in byte; the second blob contains info: compressed
    //  blob if it's compressed or original blob.
    void FilterIn(const std::vector<Blob>& blobs,
        std::vector<Blob>* outputs) override {
        CHECK_NOTNULL(outputs);
        outputs->clear();
        for (auto iter = blobs.cbegin(); iter != blobs.cend(); iter++) {
            Blob compressed_blob;
            auto compressed = try_compress(*iter, &compressed_blob);
            Blob flag_blob(sizeof(data_type) * 2);
            // compressed or not
            flag_blob.As<index_type>(0) = compressed ? 1 : 0;
            // blob size
            flag_blob.As<index_type>(1) = iter->size();
            outputs->push_back(flag_blob);
            outputs->push_back(compressed ? std::move(compressed_blob) : *iter);
        }
    }

    // Returns de-compressed blobs from input
    //  blobs compressed by function FilterIn.
    void FilterOut(const std::vector<Blob>& blobs,
        std::vector<Blob>* outputs) override {
        CHECK_NOTNULL(outputs);
        CHECK(blobs.size() % 2 == 0);
        outputs->clear();
        for (auto i = 0; i < blobs.size(); i += 2) {
            auto is_compressed = blobs[i].As<index_type>(0) == 1;
            auto size = blobs[i].As<index_type>(1);
            auto& blob = blobs[i + 1];
            outputs->push_back(is_compressed ?
                std::move(de_compress(blob, size)) : blob);
        }
    }

 protected:
    bool try_compress(const Blob& in_blob,
        Blob* out_blob) {
        CHECK_NOTNULL(out_blob);
        CHECK(sizeof(data_type) == sizeof(index_type));
        auto data_count = in_blob.size<data_type>();
        auto non_zero_count = 0;
        for (auto i = 0; i < data_count; ++i) {
            if (std::abs(in_blob.As<data_type>(i)) > clip_value) {
                ++non_zero_count;
            }
        }

        if (non_zero_count * 2 >= data_count)
            return false;

        if (non_zero_count == 0) {
            // Blob does not support empty content,
            //  fill the blob with first value

            Blob result(2 * sizeof(data_type));
            // set index
            result.As<index_type>(0) = 0;
            // set value
            result.As<data_type>(1) = in_blob.As<data_type>(0);
            *out_blob = result;
        } else {
            Blob result(non_zero_count * 2 * sizeof(data_type));
            auto result_index = 0;
            for (auto i = 0; i < data_count; ++i) {
                auto abs_value = std::abs(in_blob.As<data_type>(i));
                if (abs_value > clip_value) {
                    // set index
                    result.As<index_type>(result_index++) = i;
                    // set value
                    result.As<data_type>(result_index++) =
                        in_blob.As<data_type>(i);
                }
            }
            CHECK(result_index == non_zero_count * 2);
            *out_blob = result;
        }

        return true;
    }

    Blob de_compress(const Blob& in_blob, size_t size) {
        CHECK(sizeof(data_type) == sizeof(index_type));
        CHECK(size % sizeof(data_type) == 0);
        auto original_data_count = size / sizeof(data_type);
        Blob result(size);
        for (auto i = 0; i < original_data_count; ++i) {
            result.As<data_type>(i) = 0;
        }
        auto data_count = in_blob.size<data_type>();
        for (auto i = 0; i < data_count; i += 2) {
            auto index = in_blob.As<index_type>(i);
            auto value = in_blob.As<data_type>(i + 1);
            result.As<data_type>(index) = value;
        }

        return result;
    }

 private:
    double clip_value;
};

class OneBitsFilter : public QuantizationFilter{
};
}  // namespace multiverso

#endif  // MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_
