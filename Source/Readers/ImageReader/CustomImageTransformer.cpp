#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "Config.h"
#include "ConcStack.h"
#include "StringUtil.h"
#include "SequenceData.h"
#include "ImageUtil.h"
#include "ImageDeserializerBase.h"
#include "CustomImageTransformer.h"

namespace CNTK
{
    using namespace Microsoft::MSR::CNTK;

    inline cv::Mat cv_img_resize(const cv::Mat& cv_img_origin, const int height, const int width)
    {
        if (!cv_img_origin.data || height <= 0 || width <= 0)
        {
            return cv_img_origin;
        }
        cv::Mat cv_img;
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
        return cv_img;
    }

    inline void cv_img_hsv_color(const cv::Mat& img, const double hue, const double saturation, const double lightness)
    {
        if (hue == 0 && saturation == 1.0 && lightness == 1.0)
        {
            return;
        }
        assert(img.depth() == CV_8U || img.depth() == CV_32F || img.depth() == CV_64F);
        assert(0 <= saturation);
        assert(0 <= lightness);

        cv::Mat hsv;
        //To change saturation, we need to convert the image to HSV format first,
        //the change S channgel and convert the image back to BGR format.
        cv::cvtColor(img, hsv, CV_BGR2HSV);

        assert(hsv.rows == img.rows && hsv.cols == img.cols);
        uchar* stt = reinterpret_cast<uchar*>(hsv.data);
        uchar* end = stt + (hsv.rows * hsv.cols * img.channels());
        for (uchar* phsv = stt; phsv < end; phsv += 3)
        {
            phsv[0] = static_cast<uchar>(int(std::round(phsv[0] + hue * 0.5)) % 180);
            phsv[1] = static_cast<uchar>(std::max(0.0, std::min(phsv[1] * saturation, 255.0)));
            phsv[2] = static_cast<uchar>(std::max(0.0, std::min(phsv[2] * lightness, 255.0)));
        }
        cv::cvtColor(hsv, img, CV_HSV2BGR);
    }

    cv::Mat cv_img_pad(const cv::Mat& img, int pad_top, int pad_bottom, int pad_left, int pad_right, bool adjust, int border_type, const cv::Scalar& value)
    {
        cv::Mat dst;
        if (adjust)
        {
            pad_top = max(0, pad_top);
            pad_bottom = max(0, pad_bottom);
            pad_left = max(0, pad_left);
            pad_right = max(0, pad_right);
        }
        copyMakeBorder(img, dst, pad_top, pad_bottom, pad_left, pad_right, border_type, value);
        return dst;
    }

    cv::Mat cv_img_crop(const cv::Mat& img, cv::Rect& roi, bool adjust, bool show_only)
    {
        cv::Mat img_copy = img;
        //const Mat& img = show_only ? im : Mat(im);
        if (adjust)
        {
            if (roi.x + roi.width > img.cols)
            {
                if (show_only)
                {
                    line(img_copy, cv::Point(img.cols - 1, 0), cv::Point(img.cols - 1, img.rows - 1), cv::Scalar(0.0, 0.0, 255.0), 3);
                    //cout << "error, right :" << roi.x + roi.width << ">" << img.cols << ", left:" << roi.x << endl;
                }
                roi.width = img.cols - roi.x;
            }
            if (roi.y + roi.height > img.rows)
            {
                if (show_only)
                {
                    line(img_copy, cv::Point(roi.x, img.rows - 1), cv::Point(roi.x + roi.width, roi.y + img.rows - 1), cv::Scalar(0.0, 0.0, 255.0), 3);
                    //cout << "error, bottom:" << roi.y + roi.height << ">" << img.rows << endl;
                }
                roi.height = img.rows - roi.y;
            }
        }

        if (show_only)
        {
            rectangle(img_copy, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width, roi.y + roi.height), cv::Scalar(0.0, 255.0, 0), 1, 8, 0);
            return img_copy;
        }
        return img(roi);
    }

    inline cv::Size rotate_size(cv::Size src, double degree)
    {
        double dlen = sqrt(src.width * double(src.width) + src.height * double(src.height));
        double radian = degree * CV_PI / 180.0;
        double src_rt_rad = acos(src.width / dlen), src_rb_rad = -acos(src.width / dlen);
        double exp_rt_rad = src_rt_rad + radian, exp_rb_rad = src_rb_rad + radian;
        int exp_h = int(ceil(dlen * max(abs(sin(exp_rt_rad)), abs(sin(exp_rb_rad)))));
        int exp_w = int(ceil(dlen * max(abs(cos(exp_rt_rad)), abs(cos(exp_rb_rad)))));
        //cout << "diagonal:" << dlen << "@" << exp_rt_rad * 180 / CV_PI << ",h:" << dlen * abs(sin(exp_rt_rad)) << endl;
        //cout << "h:" << img.rows << "->" << exp_h << endl;
        //cout << "w:" << img.cols << "->" << exp_w << endl;
        return cv::Size(exp_w, exp_h);
    }

    cv::Mat cv_img_rotate(const cv::Mat& img, double degree, int pad_border_type, bool expand) //BORDER_REPLICATE)
    {
        if (degree == 0)
        {
            return cv::Mat(img);
        }
        double radian = degree * CV_PI / 180.0;
        cv::Size src_size = img.size();
        cv::Size exp_size = src_size;
        if (expand)
        {
            exp_size = rotate_size(src_size, degree);
        }

        double pad_visuable = abs(sin(radian)) * abs(cos(radian));
        int pad_h, pad_w;
        if (expand)
        {
            pad_h = max((exp_size.height - img.rows + 1) / 2, int(ceil(src_size.width * pad_visuable)));
            pad_w = max((exp_size.width - img.cols + 1) / 2, int(ceil(src_size.height * pad_visuable)));
        }
        else
        {
            double dis_h = 0.5 * (src_size.width - src_size.height); //90 or 270
            if (degree != 90 && degree != 270)
            {
                double A = tan(radian);          //,B = -1, C = 0;
                double x = src_size.width * 0.5; //,y = -src_h * 0.5;
                double dis_lt = abs(A * x + src_size.height * 0.5) / sqrt(A * A + 1);
                double dis_lb = abs(A * x - src_size.height * 0.5) / sqrt(A * A + 1);
                dis_h = max(dis_lt, dis_lb) - src_size.height * 0.5;
                //cout << "A:" << A << ",B:" << B << " dis_h:" << dis_h << endl;
            }
            pad_h = max(0, int(ceil(dis_h)));
            double dis_w = 0.5 * (src_size.height - src_size.width); //0 or 180
            if (degree != 0 && degree != 180)
            {
                double A = tan(radian + CV_PI * 0.5); //,B = -1, C = 0;
                double x = src_size.width * 0.5;
                double dis_lt = abs(A * x + src_size.height * 0.5) / sqrt(A * A + 1);
                double dis_lb = abs(A * x - src_size.height * 0.5) / sqrt(A * A + 1);
                dis_w = max(dis_lt, dis_lb) - src_size.width * 0.5;
            }
            pad_w = max(0, int(ceil(dis_w)));
        }
        //cout << "pad:" << pad_w << "," << pad_h << endl;

        const cv::Mat exp = pad_border_type < 0 ? img : cv_img_pad(img, pad_h, pad_h, pad_w, pad_w, true, pad_border_type, cv::Scalar(127.5, 127.5, 127.5));

        cv::Mat transform_m = getRotationMatrix2D(cv::Point2d(exp.cols * 0.5f, exp.rows * 0.5f), degree, 1.0); //Creating rotation matrix
        cv::Mat rot;
        warpAffine(exp, rot, transform_m, exp.size());

        int off_w = max(0, int(floor(0.5 * (exp.cols - exp_size.width))));
        int off_h = max(0, int(floor(0.5 * (exp.rows - exp_size.height))));
        cv::Rect roi = cv::Rect(off_w, off_h, exp_size.width, exp_size.height);
        //cout << angle << ":off:" << off_w << "," << off_h << endl;
        //cout << angle << ":size:" << exp_w << "," << exp_h << endl;
        return cv_img_crop(rot, roi, true, false);
    }

    CustomTransformer::CustomTransformer(const ConfigParameters& config)
        : ImageTransformerBase(config)
    {
        m_do_mirror = config(L"mirror", true);

        double _scale = config(L"scale", "1.0");
        double _scale_std = config(L"scale_std", "0.0");
        double _scale_range = config(L"scale_range", "0.0");
        m_scale.init(_scale, _scale_std, _scale_range, true);

        m_crop_size = config(L"crop_size", "0.0");

        int _width = config(L"width", 0);
        double _crop_x_offset = config(L"crop_x_offset", "0.5");
        double _crop_x = config(L"crop_x", "0.5");
        double _crop_x_std = config(L"crop_x_std", "0.5");
        double _crop_x_range = config(L"crop_x_range", "0.5");
        m_w_crop.init(m_crop_size == 0.0 ? _width : m_crop_size, _crop_x_offset, _crop_x, _crop_x_std, _crop_x_range, true);

        int _height = config(L"height", 0);
        double _crop_y_offset = config(L"crop_y_offset", "0.5");
        double _crop_y = config(L"crop_y", "0.5");
        double _crop_y_std = config(L"crop_y_std", "0.5");
        double _crop_y_range = config(L"crop_y_range", "0.5");
        m_h_crop.init(m_crop_size == 0.0 ? _height : m_crop_size, _crop_y_offset, _crop_y, _crop_y_std, _crop_y_range, true);

        double _resize = config(L"resize", "0.0");
        double _resize_std = config(L"resize_std", "0.0");
        double _resize_range = config(L"resize_range", "0.0");
        m_new_size.init(_resize, _resize_std, _resize_range);

        double _resize_width = config(L"resize_width", "0.0");
        double _resize_width_std = config(L"resize_width_std", "0.0");
        double _resize_width_range = config(L"resize_width_range", "0.0");
        m_new_width.init(_resize_width, _resize_width_std, _resize_width_range);

        double _resize_height = config(L"resize_height", "0.0");
        double _resize_height_std = config(L"resize_height_std", "0.0");
        double _resize_height_range = config(L"resize_height_range", "0.0");
        m_new_height.init(_resize_height, _resize_height_std, _resize_height_range);

        double _aspect_ratio = config(L"aspect_ratio", "1.0");
        double _aspect_ratio_std = config(L"aspect_ratio_std", "0.0");
        double _aspect_ratio_range = config(L"aspect_ratio_range", "0.0");
        m_aspect_ratio.init(_aspect_ratio, _aspect_ratio_std, _aspect_ratio_range, true);

        double _rotate_degree = config(L"rotate_degree", "0.0");
        double _rotate_degree_std = config(L"rotate_degree_std", "0.0");
        double _rotate_degree_range = config(L"rotate_degree_range", "0.0");
        m_rotate.init(_rotate_degree, _rotate_degree_std, _rotate_degree_range, true);

        double _brightness = config(L"brightness", "0.0");
        double _brightness_std = config(L"brightness_std", "0.0");
        double _brightness_range = config(L"brightness_range", "0.0");
        m_brightness.init(_brightness, _brightness_std, _brightness_range, true);

        double _hue = config(L"hue", "0.0");
        double _hue_std = config(L"hue_std", "0.0");
        double _hue_range = config(L"hue_range", "0.0");
        m_hue.init(_hue, _hue_std, _hue_range, true);

        double _saturation = config(L"saturation", "1.0");
        double _saturation_std = config(L"saturation_std", "0.0");
        double _saturation_range = config(L"saturation_range", "0.0");
        m_saturation.init(_saturation, _saturation_std, _saturation_range, true);

        double _lightness = config(L"lightness", "1.0");
        double _lightness_std = config(L"lightness_std", "0.0");
        double _lightness_range = config(L"lightness_range", "0.0");
        m_lightness.init(_lightness, _lightness_std, _lightness_range, true);

        double _bmean = config(L"B_mean", "128.0");
        m_mean_values.push_back(_bmean);
        double _gmean = config(L"G_mean", "128.0");
        m_mean_values.push_back(_gmean);
        double _rmean = config(L"R_mean", "128.0");
        m_mean_values.push_back(_rmean);
    }

    void CustomTransformer::GetRandSize(int& img_width, int& img_height, std::mt19937& rng)
    {
        int height, width;
        if (this->m_new_height.is_enabled() && this->m_new_width.is_enabled())
        {
            height = this->m_new_height.rand(img_height, rng);
            width = this->m_new_width.rand(img_width, rng);
        }
        else if (this->m_new_height.is_enabled())
        {
            height = this->m_new_height.rand(img_height, rng);
            float resize_scale = float(height) / img_height;
            float aspect_ratio = this->m_aspect_ratio.value(rng);
            width = int(img_width * resize_scale * aspect_ratio);
        }
        else if (this->m_new_width.is_enabled())
        {
            width = this->m_new_width.rand(img_width, rng);
            float resize_scale = float(width) / img_width;
            float aspect_ratio = this->m_aspect_ratio.value(rng);
            height = int(img_height * resize_scale / aspect_ratio);
        }
        else if (img_height <= img_width)
        {
            height = this->m_new_size.rand(img_height, rng);
            float resize_scale = float(height) / img_height;
            float aspect_ratio = this->m_aspect_ratio.value(rng);
            width = int(img_width * resize_scale * aspect_ratio);
        }
        else
        {
            width = this->m_new_size.rand(img_width, rng);
            float resize_scale = float(width) / img_width;
            float aspect_ratio = this->m_aspect_ratio.value(rng);
            height = int(img_height * resize_scale / aspect_ratio);
        }
        img_width = width;
        img_height = height;
    }

    cv::Mat CustomTransformer::Resize(const cv::Mat& cv_img, std::mt19937& rng)
    {
        int width = cv_img.cols, height = cv_img.rows;
        this->GetRandSize(width, height, rng);
        return cv_img_resize(cv_img, height, width);
    }

    float CustomTransformer::AbsoluteBrightness(const cv::Mat& cv_img, std::mt19937& rng)
    {
        float brightness = this->m_brightness.value(rng);
        if (brightness != 0)
        {
            // Compute mean value of the image.
            cv::Scalar imgMean = cv::sum(cv::sum(cv_img));
            // Compute beta as a fraction of the mean.
            return float(brightness * imgMean[0] / (cv_img.rows * cv_img.cols * cv_img.channels()));
        }
        return float(0.);
    }

    int Rand(int n, std::mt19937& rng)
    {
        return (rng() % n);
    }

    cv::Mat CustomTransformer::Rotate(const cv::Mat& cv_img, std::mt19937& rng)
    {
        float degree = this->m_rotate.value(rng);
        return cv_img_rotate(cv_img, degree, cv::BORDER_REPLICATE, false);
    }

    void CustomTransformer::JitterColor(const cv::Mat& cv_img, std::mt19937& rng)
    {
        float hue = this->m_hue.value(rng);
        float saturation = this->m_saturation.value(rng);
        float lightness = this->m_lightness.value(rng);
        cv_img_hsv_color(cv_img, hue, saturation, lightness);
    }

    void CustomTransformer::Apply(uint8_t, cv::Mat& mat, int indexInBatch)
    {
        auto seed = GetSeed();
        auto rng = m_rngs.at_or_create(indexInBatch, [seed](int offset) { return std::make_unique<std::mt19937>(seed + offset); });
        const cv::Mat cv_img = this->Resize(mat, *rng);
        const int crop_width = m_w_crop.size();
        const int crop_height = m_h_crop.size();
        const int img_channels = cv_img.channels();
        const int img_height = cv_img.rows;
        const int img_width = cv_img.cols;
        const int target_height = crop_height ? crop_height : img_height;
        const int target_width = crop_width ? crop_width : img_width;

        const float scale = float(this->m_scale.value(*rng));
        const float beta = this->AbsoluteBrightness(cv_img, *rng);
        const bool do_mirror = m_do_mirror && Rand(2, *rng);
        const bool has_mean_values = m_mean_values.size() > 0;

        cv::Mat cv_cropped_img = cv_img;

        int h_off = this->m_h_crop.offset(img_height, *rng); // crop_height ? Rand(img_height - crop_height + 1) : 0;
        int w_off = this->m_w_crop.offset(img_width, *rng);  // crop_width ? Rand(img_width - crop_width + 1) : 0;

        if (crop_width || crop_height)
        {
            cv::Rect roi(w_off, h_off, crop_width ? crop_width : img_width, crop_height ? crop_height : img_height);
            cv_cropped_img = cv_img(roi);
        }

        cv::Mat cv_rotated_img = this->Rotate(cv_cropped_img, *rng);

        this->JitterColor(cv_rotated_img, *rng);

        mat = cv_img_resize(mat, crop_width, crop_height);
        ConvertToFloatingPointIfRequired(mat);
        int top_index;
        for (int h = 0; h < crop_height; ++h)
        {
            const uchar* ptr = cv_rotated_img.ptr<uchar>(h);
            float* transformed_data = mat.ptr<float>(h);
            int img_index = 0;
            for (int w = 0; w < crop_width; ++w)
            {
                for (int c = 0; c < img_channels; ++c)
                {
                    if (do_mirror)
                        top_index = (crop_width - 1 - w) * img_channels + c;
                    else
                        top_index = img_index;
                    float pixel = static_cast<float>(ptr[img_index++]);
                    transformed_data[top_index] = (pixel - m_mean_values[c] + beta) * scale;
                }
            }
        }

        m_rngs.assignTo(indexInBatch, std::move(rng));
    }
}