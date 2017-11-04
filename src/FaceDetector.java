
import javax.swing.JFrame;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public class FaceDetector {

  public static void main(String arg[]) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    String window_name = "Face Detector";
    JFrame frame = new JFrame(window_name);
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setSize(400, 400);
    FaceProcessor processor = new FaceProcessor();
    MyPanel panel = new MyPanel();
    frame.setContentPane(panel);
    frame.setVisible(true);
    Mat webcamPhoto = new Mat();
    VideoCapture capture = new VideoCapture(0);
    if (capture.isOpened()) {
      while (true) {
        capture.read(webcamPhoto);
        if (!webcamPhoto.empty()) {
          frame.setSize(webcamPhoto.width() + 40, webcamPhoto.height() + 60);
          webcamPhoto = processor.detect(webcamPhoto);
          panel.MatToBufferedImage(webcamPhoto);
          panel.repaint();
        } else {
          System.out.println("No frame has been captured.");
          break;
        }
      }
    }
    return;
  }
}
