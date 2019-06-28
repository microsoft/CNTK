#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "ConcStack.h"
#include "Config.h"
#include "ImageTransformers.h"
using namespace std;

namespace CNTK
{
    class RandFloat
    {
        public:
            float mean_;
            float std_;
            float range_;
            bool max_inclusive_;
            bool is_uniform;
            float nextafter(const float val)
            {
                return boost::math::nextafter<float>(val, std::numeric_limits<float>::max());
            }

        public:
            explicit RandFloat()
                : mean_(0), std_(0), range_(0), max_inclusive_(true), is_uniform(range_ > 0 && std_ == 0)
            {
            }

            explicit RandFloat(float mean, float std, float range, bool max_inclusive)
                : mean_(mean), std_(std), range_(range), max_inclusive_(max_inclusive), is_uniform(range > 0 && std == 0)
            {
            }

            void init(float mean, float std, float range, bool max_inclusive)
            {
                mean_ = mean, std_ = std, range_ = range, max_inclusive_ = max_inclusive;
                is_uniform = range > 0 && std == 0;
            }

            inline virtual bool is_rand()
            {
                return this->range_ > 0;
            }

            inline float value(std::mt19937 &rng)
            {
                if (!this->is_rand())
                    return mean_;
                if (this->is_uniform)
                    return boost::random::uniform_real_distribution<float>(mean_ - range_, max_inclusive_ ? nextafter(mean_ + range_) : mean_ + range_)(rng);
                float x = boost::random::normal_distribution<float>(mean_, std_)(rng);
                if (x < mean_ - range_ || x >= mean_ + range_)
                    return boost::random::uniform_real_distribution<float>(mean_ - range_, max_inclusive_ ? nextafter(mean_ + range_) : mean_ + range_)(rng);

                return x;
            }

            float range()
            {
                return this->range_;
            }
    };

    struct RandSize : public RandFloat
    {
        public:
            bool is_enabled_;
            bool orginal_size_as_mean_;
            int int_range_;

        public:
            RandSize() {}
            RandSize(float mean, float std, float range)
                : RandFloat(mean + 0.5F, std, range + 0.5F, false),
                  is_enabled_(mean > 0 || range > 0),
                  orginal_size_as_mean_(mean == 0),
                  int_range_(static_cast<int>(range))
            {
            }

            void init(float mean, float std, float range)
            {
                this->mean_ = mean + 0.5F;
                this->std_ = std;
                this->range_ = range + 0.5F;
                this->max_inclusive_ = false;
                is_enabled_ = mean > 0 || range > 0;
                orginal_size_as_mean_ = mean == 0;
                int_range_ = static_cast<int>(range);
            }

            inline int mean(int img_size)
            {
                if (orginal_size_as_mean_)
                    return img_size;
                else
                    return static_cast<int>(this->mean_);
            }

            inline bool is_rand()
            {
                return this->int_range_ > 0;
            }

            inline int min(int img_size)
            {
                return std::max(1, this->mean(img_size) - this->int_range_);
            }

            inline int max(int img_size)
            {
                return std::max(1, this->mean(img_size) + this->int_range_);
            }

            inline int rand(int img_size, std::mt19937 &rng)
            {
                if (is_enabled_)
                {
                    int s = static_cast<int>(value(rng));
                    if (orginal_size_as_mean_)
                        s += img_size;
                    return std::max(1, s);
                }
                return img_size;
            }
            bool is_enabled()
            {
                return is_enabled_;
            }

            float range()
            {
                return (float)(this->int_range_);
            }
    };

    class RandCrop : public RandFloat
    {
        public:
            int crop_size;
            float offset_factor;
            bool adjust_if_out_of_boundary;

        public:
            RandCrop() {}
            RandCrop(int _crop_size, float _offset_factor, float _mean, float _std, float _range, bool _adjust_if_out_of_boundary = true)
                : RandFloat(_mean, _std, _range, true), crop_size(_crop_size), offset_factor(_offset_factor), adjust_if_out_of_boundary(_adjust_if_out_of_boundary)
            {
            }

            void init(int crop_size, float offset_factor, float mean, float std, float range, bool adjust_if_out_of_boundary = true)
            {
                mean_ = mean, std_ = std, range_ = range, max_inclusive_ = true;
                is_uniform = range > 0 && std == 0;
                this->crop_size = crop_size;
                this->offset_factor = offset_factor;
                this->adjust_if_out_of_boundary = adjust_if_out_of_boundary;
            }

            int size()
            {
                return crop_size;
            }

            inline int offset(int img_size, int crop_size, float offset_factor, float relative_pos, bool adjust_if_out_of_boundary = true)
            {
                if (crop_size == 0 || crop_size == img_size)
                {
                    return 0; // no crop
                }
                float absolute_pos = relative_pos * img_size - offset_factor * crop_size;

                int off = round(absolute_pos); //floor: uniformly distribute in [ (int)(mean - vav), (int)(mean + range) )
                if (adjust_if_out_of_boundary)
                {
                    if (off < 0)
                    {
                        return 0;
                    }
                    if (off >= img_size - crop_size)
                    {
                        return img_size - crop_size;
                    }
                }
                return off;
            }

            float nextafter(const float val)
            {
                return boost::math::nextafter<float>(val, std::numeric_limits<float>::max());
            }

            inline float value_in_range(float v_min, float v_max, std::mt19937 &rng)
            {
                if (v_min >= v_max)
                    return v_min;
                return boost::random::uniform_real_distribution<float>(v_min, v_max)(rng);
            }

            inline float relative_pos(int img_size, std::mt19937 &rng)
            {
                float x = this->value(rng);
                float legal_min = (offset_factor * crop_size) / img_size;
                float legal_max = (img_size - crop_size + offset_factor * crop_size) / img_size;
                if (x >= legal_min && x <= legal_max)
                    return x;
                float intersect_min = std::max(legal_min, mean_ - range_);
                float intersect_max = std::min(legal_max, mean_ + range_);
                if (intersect_min <= intersect_max)
                { //legal interval has intersect with desire interval
                    return value_in_range(intersect_min, intersect_max, rng);
                }
                return value_in_range(legal_min, legal_max, rng);
            }

            inline int offset(int img_size, std::mt19937 &rng)
            {
                return this->offset(img_size, this->crop_size, this->offset_factor, this->relative_pos(img_size, rng), this->adjust_if_out_of_boundary);
            }
    };

    class CustomTransformer : public ImageTransformerBase
    {
        public:
            explicit CustomTransformer(const ConfigParameters &config);

            void GetRandSize(int &img_width, int &img_height, std::mt19937 &rng);

            cv::Mat Resize(const cv::Mat &cv_img, std::mt19937 &rng);

            float AbsoluteBrightness(const cv::Mat &cv_img, std::mt19937 &rng);

            cv::Mat Rotate(const cv::Mat &cv_img, std::mt19937 &rng);

            void JitterColor(const cv::Mat &cv_img, std::mt19937 &rng);

        public:
            void Apply(uint8_t copyId, cv::Mat &mat, int indexInBatch) override;

            bool m_do_mirror;
            float *data_mean_data_;
            int config_num_mean_values_;
            vector<float> m_mean_values;
            float m_crop_size;
            RandFloat m_scale;
            RandCrop m_w_crop;
            RandCrop m_h_crop;
            RandSize m_new_size;
            RandSize m_new_width;
            RandSize m_new_height;
            RandFloat m_aspect_ratio;
            RandFloat m_rotate;
            RandFloat m_brightness;
            RandFloat m_hue;
            RandFloat m_saturation;
            RandFloat m_lightness;

            int m_maskMargin;
            int m_maskThreshold;
            float m_blacklineRate;
            int m_blacklineMargin;
            vector<int> m_blacklineMarginValue;
    };
}