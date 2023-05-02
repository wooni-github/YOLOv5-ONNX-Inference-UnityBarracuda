using Assets.Scripts;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class PhoneCamera : MonoBehaviour
{
    private bool isCamera;
    private WebCamTexture cameraTexture;
    private Texture bckgDefault;
    private static Texture2D boxOutlineTexture;
    public GameObject rects;

    List<Color> colorTag = new List<Color>();

    public RawImage bckg;
    public AspectRatioFitter fit;
    public Yolov5Detector yolov5Detector;
    public GameObject boxContainer;
    public GameObject boxPrefab;
    public GameObject background;
    public TextMeshProUGUI text;

    private int NETWORK_SIZE_X;
    private int NETWORK_SIZE_Y;

    public int CAMERA_CAPTURE_X = 640;
    public int CAMERA_CAPTURE_Y = 480;

    private int framesCount = 0;
    private float timeCount = 0.0f;
    private float refreshTime = 1.0f;
    public float fps = 0.0f;
   
    void Start()
    {
        NETWORK_SIZE_X = GameObject.Find("Detector").GetComponent<Yolov5Detector>().GetNewtorkX();  
        NETWORK_SIZE_Y = GameObject.Find("Detector").GetComponent<Yolov5Detector>().GetNewtorkY();  
        int CLASS_COUNT = GameObject.Find("Detector").GetComponent<Yolov5Detector>().GetClassCount(); 

        for (int i = 0; i < CLASS_COUNT; i++)
        {
            Color randomColor = new Color(Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), Random.Range(0.0f, 1.0f), 0.4f);
            colorTag.Add(randomColor);
        }

        bckgDefault = bckg.texture;
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            isCamera = false;
            return;
        }

        for (int i = 0; i < devices.Length; i++)
        {
            if (!devices[i].isFrontFacing)
                cameraTexture = new WebCamTexture(devices[i].name, CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y);
        }

        if (cameraTexture == null)
        {
            if (devices.Length != 0)
                cameraTexture = new WebCamTexture(devices[0].name, CAMERA_CAPTURE_X, CAMERA_CAPTURE_Y);
            else
            {
                isCamera = false;
                return;
            }
            
        }

        cameraTexture.Play();
        bckg.texture = cameraTexture;
        float ratio = ((RectTransform)background.transform).rect.width / CAMERA_CAPTURE_X;
        boxContainer.transform.localScale = new Vector2(ratio, ratio);

        isCamera = true;

    }

    // Update is called once per frame
    void Update()
    {
        if (!isCamera)
            return;

        float ratio = (float)cameraTexture.width / (float)cameraTexture.height;
        fit.aspectRatio = ratio;

        float scaleY = cameraTexture.videoVerticallyMirrored ? -1f : 1f;
        bckg.rectTransform.localScale = new Vector3(1f, scaleY, 1f);

        int orient = -cameraTexture.videoRotationAngle;
        bckg.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

        Texture texture = bckg.texture;

        WebCamTexture newTexture = (WebCamTexture)bckg.texture;
        StartCoroutine(yolov5Detector.Detect(newTexture.GetPixels32(), newTexture.width, boxes =>
        {
            Resources.UnloadUnusedAssets();
            
            foreach(Transform child in boxContainer.transform)
            {
                Destroy(child.gameObject);
            }

            for (int i = 0; i < boxes.Count; i++)
            {
                GameObject newBox = Instantiate(boxPrefab);
                newBox.name = boxes[i].Label + " " + boxes[i].Confidence;
                newBox.GetComponent<Image>().color = colorTag[boxes[i].LabelIdx];
                newBox.transform.parent = boxContainer.transform;
                newBox.transform.localPosition = new Vector3(boxes[i].Rect.x - NETWORK_SIZE_X / 2, boxes[i].Rect.y - NETWORK_SIZE_Y/ 2);
                newBox.transform.localScale = new Vector2(boxes[i].Rect.width/100, boxes[i].Rect.height/100);

                bool text = true;
                if (text)
                {
                    GameObject labelText = new GameObject("LabelText");
                    labelText.transform.parent = newBox.transform;
                    labelText.transform.localPosition = Vector3.zero;
                    Text label = labelText.AddComponent<Text>();
                    label.text = boxes[i].Label + " " + boxes[i].Confidence.ToString("F3");
                    label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                    label.fontSize = 30;
                    label.color = Color.white;
                    label.alignment = TextAnchor.MiddleCenter;

                    RectTransform labelRect = label.GetComponent<RectTransform>();
                    Vector2 pivotOffset = new Vector2(0.5f, 0.5f) - newBox.GetComponent<RectTransform>().pivot;
                    labelRect.localPosition = pivotOffset * newBox.GetComponent<RectTransform>().sizeDelta;
                }
            }
        }));

        CountFps();

    }

    private void CountFps()
    {
        if (timeCount < refreshTime)
        {
            timeCount += Time.deltaTime;
            framesCount += 1;
        }
        else
        {
            fps = (float)framesCount / timeCount;
           // Debug.Log("FPS: " + fps);
            text.text = "FPS: " + fps;
            framesCount = 0;
            timeCount = 0.0f;
        }
    }
}
