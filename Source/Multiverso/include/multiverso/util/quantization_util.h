#ifndef MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_
#define MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_

#include <multiverso/blob.h>
#include <vector>
#include <cmath>
#include <multiverso/util/log.h>

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
    explicit SparseFilter(double clip, bool skip_last_line = false) :
      clip_value_(clip), skip_option_blob_(skip_last_line){}

    ~SparseFilter() {}

    // Returns compressed blobs given input blobs.
    // - compressing will generate one more blob in result:
    //   which contained the original size of compressed row
    // - if blob compressed, the content of blob will change to:
    //   col0 ,val0, col1 , val1, ..., coln, valn
    void FilterIn(const std::vector<Blob>& blobs,
      std::vector<Blob>* outputs) override {
      CHECK_NOTNULL(outputs);
      outputs->clear();
      // since the first blob are the indicator for rows, and the last one are option
      // blob (if exist), we should not compressed that.
      outputs->push_back(blobs[0]);
      size_t data_blobs_size = skip_option_blob_? blobs.size() - 1:blobs.size();

      if (data_blobs_size > 1){ 
        Blob size_blob(sizeof(index_type) * (data_blobs_size - 1));
        outputs->push_back(size_blob);

        for (auto i = 1; i < data_blobs_size; ++i) {
          auto& blob = blobs[i];
          Blob compressed_blob;
          auto compressed = TryCompress(blob, &compressed_blob);
          // size info (compressed ? size : -1)
          size_blob.As<index_type>(i - 1) = compressed ? (index_type) blob.size() : -1;
          outputs->push_back(compressed ? std::move(compressed_blob) : blob);
        }
      }
      if (skip_option_blob_){
        //adding the original option to the output
        outputs->push_back(blobs[blobs.size() - 1]);
      }
    }

    //  Restore the blobs changed by the function FilterIn.
    void FilterOut(const std::vector<Blob>& blobs,
      std::vector<Blob>* outputs) override {
      CHECK_NOTNULL(outputs);
      // the blobs at least contain row_index blob
      CHECK(blobs.size() > 1);
      outputs->clear();
      outputs->push_back(blobs[0]);
      size_t data_blobs_size = skip_option_blob_? blobs.size() - 1:blobs.size();
      
      if (data_blobs_size > 1){
        auto& size_blob = blobs[1];
        for (auto i = 2; i < data_blobs_size; i++) {
          // size info (compressed ? size : -1)
          auto is_compressed = size_blob.As<index_type>(i - 2) >= 0;
          auto size = is_compressed ?
            size_blob.As<index_type>(i - 2) : blobs[i].size();
          auto& blob = blobs[i];
          outputs->push_back(is_compressed ?
            std::move(DeCompress(blob, size)) : blob);
        }
      }
      if (skip_option_blob_){
        //adding the original option to the output
        outputs->push_back(blobs[blobs.size() - 1]);
      }
    }

  protected:

    bool TryCompress(const Blob& in_blob,
      Blob* out_blob) {
      CHECK_NOTNULL(out_blob);
      auto data_count = in_blob.size<data_type>();
      auto non_zero_count = 0;
      for (auto i = 0; i < data_count; ++i) {
        if (std::abs(in_blob.As<data_type>(i)) > clip_value_) {
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
      }
      else {
        Blob result(non_zero_count * 2 * sizeof(data_type));
        auto result_index = 0;
        for (auto i = 0; i < data_count; ++i) {
          auto abs_value = std::abs(in_blob.As<data_type>(i));
          if (abs_value > clip_value_) {
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

    Blob DeCompress(const Blob& in_blob, size_t size) {
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
    double clip_value_;
    bool skip_option_blob_;
  };

  class OneBitsFilter : public QuantizationFilter{
  };
}  // namespace multiverso

#endif  // MULTIVERSO_UTIL_QUANTIZATION_UTIL_H_
