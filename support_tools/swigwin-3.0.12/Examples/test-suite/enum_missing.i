%module enum_missing

// Test when SWIG does not parse the enum definition
%{
enum AVPixelFormat {
    AV_PIX_FMT_NONE = -1,
    AV_PIX_FMT_YUV420P
};
enum AVPixelFormat2 {
    AV_PIX_FMT_NONE2 = -1,
    AV_PIX_FMT_YUV420P2
};

%}

%inline %{
typedef struct AVCodecCtx {
  enum AVPixelFormat pix_fmt;
  enum AVPixelFormat2 pix_fmt2;
} AVCodecCtx;

enum AVPixelFormat global_fmt;
enum AVPixelFormat2 global_fmt2;

enum AVPixelFormat use_pixel_format(enum AVPixelFormat px) {
  return px;
}
enum AVPixelFormat * use_pixel_format_ptr(enum AVPixelFormat *px) {
  return px;
}

enum AVPixelFormat2 use_pixel_format2(const enum AVPixelFormat2 px) {
  return px;
}
const enum AVPixelFormat2 * use_pixel_format_ptr2(const enum AVPixelFormat2 *px) {
  return px;
}
%}

