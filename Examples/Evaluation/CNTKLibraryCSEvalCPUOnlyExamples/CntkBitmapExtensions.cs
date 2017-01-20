//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CntkBitmapExtensions.cs -- extension methods for transforming images used in CNTK.
//
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;

namespace CNTKLibraryCSEvalExamples
{
    public static class CntkBitmapExtensions
    {
        /// <summary>
        /// Resizes an image
        /// </summary>
        /// <param name="image">The image to resize</param>
        /// <param name="width">New width in pixels</param>
        /// <param name="height">New height in pixesl</param>
        /// <param name="useHighQuality">Resize quality</param>
        /// <returns>The resized image</returns>
        public static Bitmap Resize(this Bitmap image, int width, int height, bool useHighQuality)
        {
            var newImg = new Bitmap(width, height);

            newImg.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var g = Graphics.FromImage(newImg))
            {
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                if (useHighQuality)
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                }
                else
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Default;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.Default;
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Default;
                }

                var attributes = new ImageAttributes();
                attributes.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                g.DrawImage(image, new Rectangle(0, 0, width, height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);
            }

            return newImg;
        }

        /// <summary>
        /// Extracts image pixels in CHW
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in HWC order</returns>
        public static List<float> ExtractCHW(this Bitmap image)
        {
            var features = new List<float>(image.Width * image.Height * 3);
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < image.Height; h++)
                {
                    for (int w = 0; w < image.Width; w++)
                    {
                        var pixel = image.GetPixel(w, h);
                        float v = c == 0 ? pixel.B : c == 1 ? pixel.G : pixel.R;
                        features.Add(v);
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Extracts image pixels in CHW using parallelization
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in CHW order</returns>
        public static List<float> ParallelExtractCHW(this Bitmap image)
        {
            // We use local variables to avoid contention on the image object through the multiple threads.
            int channelStride = image.Width * image.Height;
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            var features = new byte[imageWidth * imageHeight * 3];
            var bitmapData = image.LockBits(new Rectangle(0, 0, imageWidth, imageHeight), ImageLockMode.ReadOnly, image.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
            byte[] rgbValues = new byte[bytes];

            int stride = bitmapData.Stride;

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

            // The mapping depends on the pixel format
            // The mapPixel lambda will return the right color channel for the desired pixel
            Func<int, int, int, int> mapPixel = GetPixelMapper(image.PixelFormat, stride);

            // Averaged over a large number of images, these loops here execute fastest 
            // when doing Parallel.For only over c, but not over h and w.
            Parallel.For(0, 3, (int c) =>
            {
                for (int h = 0; h < imageHeight; h++)
                {
                    for (int w = 0; w < imageWidth; w++)
                    {
                        features[channelStride * c + imageWidth * h + w] = rgbValues[mapPixel(h, w, c)];
                    }
                }
            });

            image.UnlockBits(bitmapData);

            return features.Select(b => (float)b).ToList();
        }

        /// <summary>
        /// Extracts image pixels in HWC
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in HWC order</returns>
        public static List<float> ExtractHWC(this Bitmap image)
        {
            var features = new List<float>(image.Width * image.Height * 3);
            for (int w = 0; w < image.Width; w++)
            {
                for (int h = 0; h < image.Height; h++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        var pixel = image.GetPixel(w, h);
                        float v = c == 0 ? pixel.B : c == 1 ? pixel.G : pixel.R;

                        features.Add(v);
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Extracts image pixels in HWC using multiple threads
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in HWC order</returns>
        public static List<float> ParallelExtractHWC(this Bitmap image)
        {
            int heightStride = image.Width * 3;
            int widthStride = image.Height * 3;
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            var features = new byte[image.Width * image.Height * 3];
            var bitmapData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, image.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
            byte[] rgbValues = new byte[bytes];

            int stride = bitmapData.Stride;

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

            // The mapping depends on the pixel format
            // The mapPixel lambda will return the right color channel for the desired pixel
            Func<int, int, int, int> mapPixel = GetPixelMapper(image.PixelFormat, stride);

            Parallel.For(0, 3, (int c) =>
            {
                for (int h = 0; h < imageHeight; h++)
                {
                    for (int w = 0; w < imageWidth; w++)
                    {
                        features[w * widthStride + h * 3 + c] = rgbValues[mapPixel(h, w, c)];
                    };
                };
            });

            image.UnlockBits(bitmapData);

            return features.Select(b => (float)b).ToList();
        }

        /// <summary>
        /// Returns a function for extracting the R-G-B values properly from an image based on its pixel format
        /// </summary>
        /// <param name="pixelFormat">The image's pixel format</param>
        /// <param name="heightStride">The stride (row byte count)</param>
        /// <returns>A function with signature (height, width, channel) returning the corresponding color value</returns>
        private static Func<int, int, int, int> GetPixelMapper(PixelFormat pixelFormat, int heightStride)
        {
            switch (pixelFormat)
            {
                case PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * heightStride + w * 4 + c;  // bytes are B-G-R-A
                case PixelFormat.Format24bppRgb:
                default:
                    return (h, w, c) => h * heightStride + w * 3 + c;  // bytes are B-G-R
            }
        }
    }
}
