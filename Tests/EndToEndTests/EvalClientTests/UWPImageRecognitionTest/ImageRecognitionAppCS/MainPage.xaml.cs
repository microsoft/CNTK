//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using ImageRecognitionUtils;

namespace ImageRecognitionCS
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page, IImageRecognizerConsole
    {
        ImageRecognizerDriver driver;

        void IImageRecognizerConsole.ShowText(string text)
        {
            this.text.Text += text;
        }

        void IImageRecognizerConsole.ShowProgress(bool progress)
        {
            this.progress.IsActive = progress;
            this.cntkPickModel.IsEnabled = !progress;
            this.cntkPickImage.IsEnabled = !progress;
        }

        public MainPage()
        {
            this.InitializeComponent();
            this.driver = new ImageRecognizerDriver(this);
            this.cntkPickModel.IsEnabled = true;
            this.cntkPickImage.IsEnabled = false;
        }

        private async void cntkPickImage_Click(object sender, RoutedEventArgs e)
        {
            await driver.PickAndRecognizeImageAsync();
        }

        private async void cntkPickModel_Click(object sender, RoutedEventArgs e)
        {
            await driver.LoadModelAsync();
            this.cntkPickModel.IsEnabled = false;
            this.cntkPickImage.IsEnabled = true;
        }

    }
}
