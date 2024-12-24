using UnityEngine;
using System.IO;

public class CameraCapture : MonoBehaviour
{
    private Camera mainCamera;
    private string savePath = "D:/JHU/CS/Research/GenexWorld/ProcessData/pics/UnityCapture/CapturedImage.png";
    private bool hasCaptured = false;
    
    // 4K resolution constants
    private const int RESOLUTION_WIDTH = 3840;
    private const int RESOLUTION_HEIGHT = 2160;

    void Start()
    {
        Debug.Log("CameraCapture script started");
        
        // Get the main camera if not assigned
        mainCamera = GetComponent<Camera>();
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
            Debug.Log("Using Camera.main");
        }
        
        if (mainCamera == null)
        {
            Debug.LogError("No camera found! Please attach this script to a camera or make sure there's a camera tagged as MainCamera");
            return;
        }
        
        Debug.Log($"Camera found: {mainCamera.name}");
        Debug.Log($"Capture resolution set to: {RESOLUTION_WIDTH}x{RESOLUTION_HEIGHT}");
        
        // Capture automatically after 1 second to ensure everything is initialized
        Invoke("CaptureImage", 1f);
    }

    void Update()
    {
        // Also capture when Space key is pressed
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space key pressed - capturing image");
            CaptureImage();
        }
    }

    public void CaptureImage()
    {
        Debug.Log("Starting 4K image capture...");
        
        if (mainCamera == null)
        {
            Debug.LogError("Camera is null!");
            return;
        }

        try
        {
            // Create a new RenderTexture at 4K resolution
            Debug.Log($"Creating RenderTexture with 4K dimensions: {RESOLUTION_WIDTH}x{RESOLUTION_HEIGHT}");
            RenderTexture rt = new RenderTexture(RESOLUTION_WIDTH, RESOLUTION_HEIGHT, 24);
            rt.antiAliasing = 8; // Add anti-aliasing for better quality
            
            // Store the current render texture and camera properties
            RenderTexture previousRT = RenderTexture.active;
            float previousAspect = mainCamera.aspect;
            
            // Set the camera to render at 4K aspect ratio
            mainCamera.aspect = (float)RESOLUTION_WIDTH / RESOLUTION_HEIGHT;
            
            // Set the camera to render to our texture
            mainCamera.targetTexture = rt;
            RenderTexture.active = rt;
            
            // Render the camera's view
            mainCamera.Render();
            Debug.Log("Camera rendered to 4K texture");
            
            // Create a new texture and read the active render texture into it
            Texture2D screenshot = new Texture2D(RESOLUTION_WIDTH, RESOLUTION_HEIGHT, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, RESOLUTION_WIDTH, RESOLUTION_HEIGHT), 0, 0);
            screenshot.Apply();
            
            // Clean up and restore camera properties
            mainCamera.targetTexture = null;
            mainCamera.aspect = previousAspect;
            RenderTexture.active = previousRT;
            Destroy(rt);
            
            // Create the directory if it doesn't exist
            string directory = Path.GetDirectoryName(savePath);
            if (!Directory.Exists(directory))
            {
                Debug.Log($"Creating directory: {directory}");
                Directory.CreateDirectory(directory);
            }
            
            // Save to file
            byte[] bytes = screenshot.EncodeToPNG();
            File.WriteAllBytes(savePath, bytes);
            Debug.Log($"4K screenshot successfully saved to: {savePath}");
            
            // Clean up
            Destroy(screenshot);
            hasCaptured = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error during capture: {e.Message}\nStack trace: {e.StackTrace}");
        }
    }
}