from models import Detector
import cv2


if __name__ == '__main__':
    detector = Detector(cnn_version=0)
    # test2.mp4 is a web cam video saved locally
    capture = cv2.VideoCapture('test2.mp4')
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    while True:
        check, frame = capture.read()
        detector.webcam_predict(frame, frame_width, frame_height)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
