using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace Digitz
{
    class DigitRecognizer
    {
        private readonly MLContext _context = new MLContext();
        private ITransformer _model;
        private PredictionEngine<MNistInput, MNistOutput> _engine;

        public DigitRecognizer()
        {
            LoadModel("MNIST-original.zip");
            //LoadModel("MNIST-retrained.zip");
            //LoadModel("MNIST-retrainedWithHash.zip");
        }

        /// <summary>
        /// Returns the digit represented by the image.
        /// </summary>
        public List<DigitResult> Evaluate(Bitmap image)
        {
            float[] imageData = ConvertImageToTensorData(image);

            return Evaluate(imageData);
        }

        /// <summary>
        /// Converts the image into the expected data for the MNIST model.
        /// </summary>
        private float[] ConvertImageToTensorData(Bitmap image)
        {
            int width = 28;
            int height = 28;
            image = ResizeImage(image, new Size(width, height));

            float[] imageData = new float[width * height];

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Color color = image.GetPixel(x, y);
                    float pixelValue = (color.R + color.G + color.B) / 3;

                    // Turn to black background and white digit like MNIST model expects
                    imageData[x + (y * height)] = (255 - pixelValue);
                }
            }

            return imageData;
        }

        internal void Record(Bitmap image)
        {
            float[] imageData = ConvertImageToTensorData(image);

            File.AppendAllText(@"..\..\..\..\Training\input\MNISTRecords.txt", string.Join(",", imageData) + Environment.NewLine);
        }

        private List<DigitResult> Evaluate(float[] imageData)
        {
            try
            {
                var prediction = _engine.Predict(new MNistInput()
                {
                    Placeholder = imageData
                });

                List<DigitResult> results = new List<DigitResult>(10);
                results.Add(new DigitResult() { Digit = (int)prediction.PredictedLabel });

                // sort so the highest confidence is first
                results = results.OrderByDescending(r => r.Confidence).ToList();

                return results;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return new List<DigitResult>();
            }
        }

        private void LoadModel(string modelFilePath)
        {
            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException(
                    modelFilePath,
                    $"Error: The model '{modelFilePath}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/ConvNet to create the model.");
            }

            _model = _context.Model.Load(modelFilePath, out _);
            _engine = _context.Model.CreatePredictionEngine<MNistInput, MNistOutput>(_model);
        }

        private static Bitmap ResizeImage(Bitmap imgToResize, Size size)
        {
            Bitmap b = new Bitmap(size.Width, size.Height);
            using (Graphics g = Graphics.FromImage(b))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
            }
            return b;
        }
    }

    public class DigitResult
    {
        public int Digit { get; set; }
        public float Confidence { get; set; }
    }

    class MNistInput
    {
        public long Label { get; set; }
        [VectorType(784)]
        public float[] Placeholder { get; set; }
    }

    class MNistOutput
    {
        public long PredictedLabel { get; set; }
    }
}
