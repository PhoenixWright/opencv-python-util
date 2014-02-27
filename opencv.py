import logging
import os.path # for checking for local files
import numpy # image processing support
import cv2 # image analyzing via OpenCV

# logging
log = logging.getLogger('opencv')
log.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(message)s')
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


def is_png(data):
    """True if data is the first 8 bytes of a PNG file."""
    return data[:8] == '\x89PNG\x0d\x0a\x1a\x0a'


def opencvify_image(image):
    if os.path.isfile(image):
        return cv2.imread(image, 0)

    if is_png(image):
        # the goal here is to make it seem like a buffer was loaded from a file
        img_array = numpy.asarray(bytearray(png1), dtype=numpy.uint8)
        return cv2.imdecode(img_array, 0)

    return None


def opencv_open_file(path):
    return cv2.imread(path, 0)


def opencv_feature_match(png1, png2, feature_detector='SIFT', threshold=0.75):
    """Given two images in png format in memory, compare them and return the
    number of matches and a third image.

    :param feature_detector: "ORB", "FAST", "BRISK"

    """
    img1 = opencvify_image(png1)
    img2 = opencvify_image(png2)

    detector = cv2.FeatureDetector_create(feature_detector)
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    sift = cv2.SIFT()
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    log.debug('{} matches found.'.format(len(matches)))

    # sort matches in order of distance
    matches = sorted(matches, key = lambda x:x.distance)

    """des1 = numpy.asarray(des1, dtype=numpy.float32)
    des2 = numpy.asarray(des2, dtype=numpy.float32)

    # flann settings
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)"""

    """bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) (FOR ORB)"""

    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    log.debug('{} matches were within the threshold.'.format(len(good)))

    # draw matches and output to a third image
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)
    return_value, image_analysis_array = cv2.imencode('.png', img3)

    """output = open('test.png', 'w')
    output.write(image_analysis_array.tostring())
    output.close()"""

    return (len(matches), image_analysis_array.tostring())

