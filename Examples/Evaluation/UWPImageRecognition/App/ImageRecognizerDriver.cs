//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using ImageRecognizerLib;
using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace ImageRecognitionUtils
{
    interface IImageRecognizerConsole
    {
        void ShowText(string text);
        void ShowProgress(bool progress);
    }

    class ImageRecognizerDriver
    {
        IImageRecognizerConsole console;
        CNTKImageRecognizer cntkRecognizer;

        public ImageRecognizerDriver(IImageRecognizerConsole console)
        {
            this.console = console;
        }

        public async Task LoadModelAsync()
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = PickerLocationId.DocumentsLibrary;
            picker.FileTypeFilter.Add(".model");
            var pickedFile = await picker.PickSingleFileAsync();
            if (pickedFile != null)
            {
                // The file cannot be read directly from the DocumentsLibrary, so copy the file into the local app folder
                var localFolder = Windows.Storage.ApplicationData.Current.LocalFolder;
                var localFile = await pickedFile.CopyAsync(localFolder, pickedFile.Name, NameCollisionOption.ReplaceExisting);

                var sw = Stopwatch.StartNew();
                console.ShowText("Loading CNTK Model... ");
                console.ShowProgress(true);

                try
                {
                    var path = localFile.Path;
                    this.cntkRecognizer = CNTKImageRecognizer.Create(path, "Assets\\imagenet1000_clsid.txt");
                    sw.Stop();
                    console.ShowText($"Elapsed time: {sw.ElapsedMilliseconds} ms");
                }
                catch (Exception ex)
                {
                    console.ShowText($"error: {ex.Message}");
                    sw.Stop();
                }
                console.ShowProgress(false);
            }
        }

        public async Task PickAndRecognizeImageAsync()
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".jpg");
            var file = await picker.PickSingleFileAsync();
            if (file != null)
            {
                console.ShowProgress(true);
                try
                {
                    await RecognizeFile(file);
                }
                finally
                {
                    console.ShowProgress(false);
                }
            }
        }

        private async Task RecognizeFile(StorageFile file)
        {
            var fileStream = await file.OpenAsync(Windows.Storage.FileAccessMode.Read);

            var decoder = await BitmapDecoder.CreateAsync(fileStream);

            uint sHeight = cntkRecognizer.GetRequiredHeight();
            uint sWidth = cntkRecognizer.GetRequiredWidth();

            BitmapTransform transform = new BitmapTransform()
            {
                ScaledHeight = sHeight,
                ScaledWidth = sWidth
            };

            PixelDataProvider pixelData = await decoder.GetPixelDataAsync(
                BitmapPixelFormat.Rgba8,
                BitmapAlphaMode.Straight,
                transform,
                ExifOrientationMode.RespectExifOrientation,
                ColorManagementMode.DoNotColorManage);

            var data = pixelData.DetachPixelData();
            var sw = Stopwatch.StartNew();

            string objectName = "?";
            try
            {
                objectName = await cntkRecognizer.RecognizeObjectAsync(data);
            }
            catch
            {
                objectName = "error";
            }

            sw.Stop();

            console.ShowText(String.Format("\n{0} -> {1}. Elapsed time: {2} ms", file.Name, objectName, sw.ElapsedMilliseconds));
        }
    }
}