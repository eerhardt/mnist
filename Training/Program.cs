using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.IO;

namespace TFRetrain
{
    class Program
    {
        static readonly string model_location = "mnist_conv_model";
        static readonly MLContext mlContext = new MLContext(seed: 1);
        static readonly TextLoader loader = mlContext.Data.CreateTextLoader(columns: new[]
            {
                new TextLoader.Column("Label", DataKind.Int64, 0),
                new TextLoader.Column("Placeholder", DataKind.Single, new []{ new TextLoader.Range(1, 784) })
            },
            separatorChar: ',');

        static void Main(string[] args)
        {
            //CreateOriginalTFModel();
            RetrainTFModel();
        }

        private static void CreateOriginalTFModel()
        {
            IEstimator<ITransformer> pipe = mlContext.Transforms.NormalizeMinMax("Features", "Placeholder");
            pipe = pipe
                .Append(mlContext.Model.LoadTensorFlowModel(model_location).ScoreTensorFlowModel(
                    inputColumnNames: new[] { "Features" },
                    outputColumnNames: new[] { "Prediction" }
                    ));
            pipe = pipe
                .Append(mlContext.Transforms.CopyColumns("Features", "Prediction"));
            pipe = pipe
                .Append(mlContext.Transforms.Conversion.MapValueToKey("KeyLabel", "Label", maximumNumberOfKeys: 10));
            pipe = pipe
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("KeyLabel", "Features"));
            pipe = pipe
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            EvaluateAndSaveModel(pipe, "MNIST-original");
        }

        private static void RetrainTFModel()
        {
            IEstimator<ITransformer> pipe = mlContext.Transforms.NormalizeMinMax("Features", "Placeholder");
            pipe = pipe
                .Append(mlContext.Model.LoadTensorFlowModel(model_location).RetrainTensorFlowModel(
                    inputColumnNames: new[] { "Features" },
                    outputColumnNames: new[] { "Prediction" },
                    labelColumnName: "Label",
                    tensorFlowLabel: "Label",
                    optimizationOperation: "MomentumOp",
                    lossOperation: "Loss",
                    epoch: 10,
                    learningRateOperation: "learning_rate",
                    learningRate: 0.005f,
                    batchSize: 20));
            pipe = pipe
                .Append(mlContext.Transforms.CopyColumns("Features", "Prediction"));
            pipe = pipe
                .Append(mlContext.Transforms.Conversion.MapValueToKey("KeyLabel", "Label", maximumNumberOfKeys: 10));
            pipe = pipe
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("KeyLabel", "Features"));
            pipe = pipe
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            EvaluateAndSaveModel(pipe, "MNIST-retrained");
            //EvaluateAndSaveModel(pipe, "MNIST-retrainedWithHash");
        }

        private static void EvaluateAndSaveModel(IEstimator<ITransformer> pipe, string modelFileName)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            var trainData = mlContext.Data.Cache(loader.Load(@"..\..\..\input\Train-28x28_cntk_text.txt"), "Label", "Placeholder");
            var testData = loader.Load(@"..\..\..\input\Test-28x28_cntk_text.txt");

            ITransformer model = pipe.Fit(trainData);

            var testResults = model.Transform(testData);

            var metrics = mlContext.MulticlassClassification.Evaluate(testResults, labelColumnName: "KeyLabel");
            Console.WriteLine($"Training and evaluation time: {watch.Elapsed}");
            Console.WriteLine(metrics.MacroAccuracy);
            Console.WriteLine(metrics.MicroAccuracy);

            Directory.CreateDirectory(@"..\..\..\output\");
            mlContext.Model.Save(model, null, @$"..\..\..\output\{modelFileName}.zip");
        }
    }
}
