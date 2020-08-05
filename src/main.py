import cv2

from face_detection import FaceDetection
from input_feeder import InputFeeder


def process(pipeline, frame):
    for op in pipeline:
        result = op(frame)

    return result


def draw_bounding_boxes(coords, frame):
    height, width, _ = frame.shape
    x_min = int(coords[0] * width)
    y_min = int(coords[1] * height)
    x_max = int(coords[2] * width)
    y_max = int(coords[3] * height)

    frame = cv2.rectangle(frame,
                          (x_min, y_min),
                          (x_max, y_max),
                          (255, 0, 0),
                          1)

    return frame


def run():
    cv2.namedWindow("video")

    face = FaceDetection("face-detection-adas-0001")
    face.load_model()

    feed = InputFeeder('cam')
    feed.load_data()

    pipeline = [face.predict]

    for batch in feed.next_batch():
        result = process(pipeline, batch)
        batch = draw_bounding_boxes(result, batch)

        cv2.imshow("video", batch)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(0)

    feed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # TODO: Add arguments
    run()
