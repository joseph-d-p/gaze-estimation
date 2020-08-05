import cv2

from input_feeder import InputFeeder


def process(frame):
    # TODO: use inference pipeline
    pass


def run():
    cv2.namedWindow("video")
    feed = InputFeeder('cam')
    feed.load_data()

    for batch in feed.next_batch():
        process(batch)

        cv2.imshow("video", batch)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(0)

    feed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # TODO: Add arguments
    run()
