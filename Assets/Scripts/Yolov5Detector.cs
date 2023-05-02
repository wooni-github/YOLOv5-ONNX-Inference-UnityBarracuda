using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Unity.Barracuda;
using UnityEngine;

namespace Assets.Scripts
{
    public class Yolov5Detector : MonoBehaviour, Detector
    {
        private string INPUT_NAME;
        private string OUTPUT_NAME;

        private int NETWORK_SIZE_X = 416;
        private int NETWORK_SIZE_Y = 416;

        private int CLASS_COUNT = 40;
        private int OUTPUT_ROWS = 10647;
        
        public float MINIMUM_CONFIDENCE = 0.25f;
        public float MINIMUM_IOU_BBOX = 0.45f;

        public int OBJECTS_LIMIT = 20;
        public NNModel modelFile;
        private IWorker worker;

        private const int IMAGE_MEAN = 0;
        private const float IMAGE_STD = 255.0F;

        private List<string> classNames = new List<string>();

        private void ONNXParse(string inputString)
        {
            string pattern = @"'([^']*)'";
            MatchCollection matches = Regex.Matches(inputString, pattern);
            foreach (Match match in matches)
            {
                classNames.Add(match.Groups[1].Value);
            }
            CLASS_COUNT = classNames.Count();
        }

        public void Start()
        {
            var model = ModelLoader.Load(this.modelFile);
            //this.worker = GraphicsWorker.GetWorker(model);
            this.worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model);

            int[] inputShape = model.inputs[0].shape;
            INPUT_NAME = model.inputs[0].name;
            OUTPUT_NAME = model.outputs[0];

            var dictionary = (Dictionary<string, string>)model.Metadata;
            ONNXParse(model.Metadata["names"]);
            NETWORK_SIZE_X = inputShape[6];
            NETWORK_SIZE_Y = inputShape[5];            
        }

        public int GetNewtorkX()
        {
            return NETWORK_SIZE_X;
        }
        public int GetNewtorkY()
        {
            return NETWORK_SIZE_Y;
        }
        public int GetClassCount()
        {
            return CLASS_COUNT;
        }

        public IEnumerator Detect(Color32[] picture, int width, System.Action<IList<BoundingBox>> callback)
        {
            using (var tensor = TransformInput(picture, NETWORK_SIZE_X, NETWORK_SIZE_Y, width))
            {
                var inputs = new Dictionary<string, Tensor>();
                inputs.Add(INPUT_NAME, tensor);
                yield return StartCoroutine(worker.StartManualSchedule(inputs));
                
                var output = worker.PeekOutput(OUTPUT_NAME);
                OUTPUT_ROWS = output.kernelCount;
                var results = ParseYoloV5Output(output, MINIMUM_CONFIDENCE);
                var boxes = FilterBoundingBoxes(results, OBJECTS_LIMIT, MINIMUM_IOU_BBOX);
                callback(boxes);
            }
        }

        private static Tensor TransformInput(Color32[] pic, int width, int height, int requestedWidth)
        {
            float[] floatValues = new float[width * height * 3];
            int beginning = (((pic.Length / requestedWidth) - height) * requestedWidth) / 2;
            int leftOffset = (requestedWidth - width) / 2;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    var color = pic[beginning + leftOffset + j];

                    floatValues[(i * width + j) * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
                    floatValues[(i * width + j) * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
                    floatValues[(i * width + j) * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
                }
                beginning += requestedWidth;
            }
            return new Tensor(1, height, width, 3, floatValues);
        }
        private IList<BoundingBox> ParseYoloV5Output(Tensor tensor, float MINIMUM_CONFIDENCE)
        {
            var boxes = new List<BoundingBox>();

            for (int i = 0; i < OUTPUT_ROWS; i++)
            {
                float confidence = GetConfidence(tensor, i);
                if (confidence < MINIMUM_CONFIDENCE)
                    continue;

                BoundingBoxDimensions dimensions = ExtractBoundingBoxDimensionsYolov5(tensor, i);
                (int classIdx, float maxClass) = GetClassIdx(tensor, i);

                float maxScore = confidence * maxClass;

                if (maxScore < MINIMUM_CONFIDENCE)
                    continue;

                boxes.Add(new BoundingBox
                {
                    Dimensions = MapBoundingBoxToCell(dimensions),
                    Confidence = confidence,
                    Label = classNames[classIdx],
                    LabelIdx = classIdx
                });
            }

            return boxes;
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensionsYolov5(Tensor tensor, int row)
        {
            return new BoundingBoxDimensions
            {
                X = tensor[0, 0, 0, row],
                Y = tensor[0, 0, 1, row],
                Width = tensor[0, 0, 2, row],
                Height = tensor[0, 0, 3, row]
            };
        }

        private float GetConfidence(Tensor tensor, int row)
        {
            float tConf = tensor[0, 0, 4, row];
            return Sigmoid(tConf);
        }

        private ValueTuple<int, float> GetClassIdx(Tensor tensor, int row)
        {
            int classIdx = 0;

            float maxConf = tensor[0, 0, 5, row];

            for (int i = 0; i < CLASS_COUNT; i++)
            {
                if (tensor[0, 0, 5 + i, row] > maxConf)
                {
                    maxConf = tensor[0, 0, 5 + i, row];
                    classIdx = i;
                }
            }
            return (classIdx, maxConf);
        }

        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);

            return k / (1.0f + k);
        }

        private BoundingBoxDimensions MapBoundingBoxToCell(BoundingBoxDimensions boxDimensions)
        {
            return new BoundingBoxDimensions
            {
                X = (boxDimensions.X),
                Y = (boxDimensions.Y),
                Width = boxDimensions.Width,
                Height = boxDimensions.Height,
            };
        }

        private IList<BoundingBox> FilterBoundingBoxes(IList<BoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
            {
                isActiveBoxes[i] = true;
            }

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            var results = new List<BoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

        private float IntersectionOverUnion(Rect boundingBoxA, Rect boundingBoxB)
        {
            var areaA = boundingBoxA.width * boundingBoxA.height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.width * boundingBoxB.height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.xMin, boundingBoxB.xMin);
            var minY = Math.Max(boundingBoxA.yMin, boundingBoxB.yMin);
            var maxX = Math.Min(boundingBoxA.xMax, boundingBoxB.xMax);
            var maxY = Math.Min(boundingBoxA.yMax, boundingBoxB.yMax);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }
    }
}
