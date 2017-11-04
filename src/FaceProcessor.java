import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceProcessor {

  private CascadeClassifier faceCascade;
  private CascadeClassifier eyesCascade;
  private CascadeClassifier mouthCascade;

  public FaceProcessor() {
    faceCascade = new CascadeClassifier(
        FaceDetector.class.getResource("haarcascade_frontalface_alt.xml")
            .getPath());
    eyesCascade = new CascadeClassifier(
        FaceDetector.class.getResource("haarcascade_eye_tree_eyeglasses.xml")
            .getPath());
    mouthCascade = new CascadeClassifier(
        FaceDetector.class.getResource("Mouth.xml").getPath());
    if (faceCascade.empty()) {
      System.out.println("Face cascade failed to load.");
      return;
    } else {
      System.out.println("Face cascade loaded successfully.");
    }

    if (eyesCascade.empty()) {
      System.out.println("Eyes cascade failed to load.");
    } else {
      System.out.println("Eyes cascade loaded successfully.");
    }

    if (mouthCascade.empty()) {
      System.out.println("Mouth cascade failed to load.");
    } else {
      System.out.println("Mouth cascade loaded successfully.");
    }

  }

  public Mat detect(Mat inputframe) {
    Mat mRgba = new Mat();
    Mat mGrey = new Mat();
    MatOfRect faces = new MatOfRect();
    inputframe.copyTo(mRgba);
    inputframe.copyTo(mGrey);
    Imgproc.cvtColor(mRgba, mGrey, Imgproc.COLOR_BGR2GRAY);
    Imgproc.equalizeHist(mGrey, mGrey);
    faceCascade.detectMultiScale(mGrey, faces);
    System.out
        .println(String.format("Detected %s face(s)", faces.toArray().length));
    faceCascade.detectMultiScale(mGrey, faces, 1.1, 5, 0, new Size(30, 30),
        new Size());
    Rect[] facesArray = faces.toArray();

    for (int i = 0; i < facesArray.length; i++) {
      Point centre1 = new Point(facesArray[i].x + facesArray[i].width * 0.5,
          facesArray[i].y + facesArray[i].height * 0.5);
      Core.ellipse(mRgba, centre1,
          new Size(facesArray[i].width * 0.5, facesArray[i].height * 0.5), 0, 0,
          360,
          new Scalar(192, 202, 235), 3, 8, 0);

      Mat faceROI = mGrey.submat(facesArray[i]);
      MatOfRect eyes = new MatOfRect();

      eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 5, 0, new Size(30, 30),
          new Size());

      Rect[] eyesArray = eyes.toArray();

      for (int j = 0; j < eyesArray.length; j++) {
        Point centre2 = new Point(
            facesArray[i].x + eyesArray[j].x + eyesArray[j].width * 0.5,
            facesArray[i].y + eyesArray[j].y + eyesArray[j].height * 0.5);
        int radius = (int) Math
            .round((eyesArray[j].width + eyesArray[j].height) * 0.25);
        Core.circle(mRgba, centre2, radius, new Scalar(255, 202, 121), 3, 8, 0);
      }

      MatOfRect mouth = new MatOfRect();

      facesArray[i].height = (int) Math.round(facesArray[i].height * 0.5);
      facesArray[i].y = facesArray[i].y + facesArray[i].height;

      faceROI = mGrey.submat(facesArray[i]);

      mouthCascade
          .detectMultiScale(faceROI, mouth, 1.1, 80, 0, new Size(30, 30),
              new Size());

      Rect[] mouthArray = mouth.toArray();

      for (int k = 0; k < mouthArray.length; k++) {
        Point centre3 = new Point(
            facesArray[i].x + mouthArray[k].x + mouthArray[k].width * 0.5,
            facesArray[i].y + mouthArray[k].y + mouthArray[k].height * 0.5);
        Core.ellipse(mRgba, centre3,
            new Size(mouthArray[k].width * 0.5, mouthArray[k].height * 0.5), 0,
            0, 360,
            new Scalar(177, 138, 255), 3, 8, 0);
      }

    }
    return mRgba;
  }
}
