import pdb

import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
import pdb
import math


#   INSTRUCTION MANUAL
#  CODE NEEDS TWO INPUTS: AN IMAGE SOURCE AND THE LENGTH OF THE REFERENCE OBJECT
#  THE REFERENCE OBJECT CAN BE A SQUARE OR A CIRCLE
#  THE REFERENCE OBJECT SHOULD BE PLACED LEFTMOST ON THE IMAGE


# HELPER FUNCTIONS
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# MAIN CLASS
class Vision:
    def __init__(self, length=None, path=None, img=None, threshold=5000):
        if img is None:
            self.original = cv.imread(path)
            self.img = cv.imread(path)
        else:
            self.original = img
            self.img = img
        self.contours = None
        self.pixelsPerMetric = None
        self.boundingBox = []
        self.bbContours = []
        self.contourThreshold = threshold
        self.length = length
        self.windowName = "Akasha Vision"
        cv.namedWindow(self.windowName)
        self.tb1Value = None
        self.bottomLine = None
        self.radius = None

    def updateImage(self, img):
        self.img = img

    def clrFilter(self, range1, range2):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(self.img, range1, range2)
        imask = mask > 0
        green = np.zeros_like(self.img, np.uint8)
        green[imask] = self.img[imask]
        self.img = green

    def on_trackbar_1(self, val):
        self.tb1Value = val
        ret, img = cv.threshold(self.img.copy(), val, 255, cv.THRESH_BINARY)
        cv.imshow("Akasha Vision", img)

    def rotateClockwise(self, reps):
        for i in range(reps):
            self.img = cv.rotate(self.img, 0)
            self.original = self.img

    def resizeAspect(self, width=None, height=None, inter=cv.INTER_AREA):
        dim = None
        (h, w) = self.img.shape[:2]

        if width is None and height is None:
            return self.img
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        self.img = cv.resize(self.img, dim, interpolation=inter)
        self.original = self.img

    def show(self, wait=True, src=None):
        cv.imshow("Akasha Vision", self.img)
        if wait:
            cv.waitKey(0)

    def findContours(self):
        # self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # self.img = cv.GaussianBlur(self.img, (7, 7), 0)
        #
        # self.img = cv.Canny(self.img, 0, 100)
        self.img = cv.dilate(self.img, None, iterations=1)
        self.img = cv.erode(self.img, None, iterations=1)

        self.contours = cv.findContours(self.img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(self.contours)
        (self.contours, _) = contours.sort_contours(self.contours)

    def resetImage(self):
        self.img = self.original

    def drawAllContours(self):
        self.resetImage()
        for contour in self.contours:
            if cv.contourArea(contour) < self.contourThreshold:
                continue

            bb = cv.minAreaRect(contour)
            bb = cv.boxPoints(bb)
            bb = np.array(bb, dtype="int")
            bb = perspective.order_points(bb)
            if abs(bb[0][0] - bb[3][0]) > 1:
                continue

            # self.boundingBox.append(bb)

            self.img = self.original
            cv.drawContours(self.img, [bb.astype("int")], -1, (255, 255, 0), 2)
            #
            # DRAW RED DOTS AT THE CORNERS OF THE BOUNDING BOX
            for (x, y) in bb:
                cv.circle(self.img, (int(x), int(y)), 5, (0, 0, 255), -1)

    def computeAllDistance(self):
        for contour in self.contours:
            if cv.contourArea(contour) < self.contourThreshold:
                continue

            self.boundingBox = cv.minAreaRect(contour)
            self.boundingBox = cv.boxPoints(self.boundingBox)
            self.boundingBox = np.array(self.boundingBox, dtype="int")
            self.boundingBox = perspective.order_points(self.boundingBox)

            (tl, tr, br, bl) = self.boundingBox
            (tltrX, tltrY) = midpoint(tl, tr)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = dist.euclidean(tl, bl)
            dB = dist.euclidean(tl, tr)

            if self.pixelsPerMetric is None:
                self.pixelsPerMetric = dB / self.length

            dimA = dA / self.pixelsPerMetric
            dimB = dB / self.pixelsPerMetric
            cv.putText(self.img, "{:.1f}mm".format(dimB),
                       (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
                       0.65, (255, 255, 255), 2)
            cv.putText(self.img, "{:.1f}mm".format(dimA),
                       (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
                       0.65, (255, 255, 255), 2)

    def blur(self, val=0, img=None):
        if img is None:
            self.img = cv.GaussianBlur(self.img, (val, val), 0)
        else:
            img = cv.GaussianBlur(img, (val, val), 0)
            return img

    def gray(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def edge(self):
        self.img = cv.Canny(self.img, 40, 90)

    def getContour(self):
        self.contours = cv.findContours(self.img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = imutils.grab_contours(self.contours)
        (self.contours, _) = contours.sort_contours(self.contours)
        print("Found " + str(len(self.contours)) + " contours.")

    # GET ONLY LARGE AND RIGHT-ANGLE BOUNDING BOXES
    def getBoundingBox(self):
        for contour in self.contours:
            if cv.contourArea(contour) < self.contourThreshold:
                continue

            bb = cv.minAreaRect(contour)
            bb = cv.boxPoints(bb)
            bb = np.array(bb, dtype="int")
            bb = perspective.order_points(bb)
            if abs(bb[0][0] - bb[3][0]) > 1:
                continue

            self.boundingBox.append(bb)
            self.bbContours.append(contour)

    def scanForBottomCorner(self):
        bottleContour = self.bbContours[0]
        prev_cnt = (9999, 9999)
        for i in range(bottleContour.shape[0]):
            if (i + 5) > bottleContour.shape[0]:
                continue
            p1 = bottleContour[i][0]
            p2 = bottleContour[i + 4][0]
            if (p2[0] - p1[0]) == 0:
                continue
            gradient = (p2[1] - p1[1]) / (p2[0] - p1[0])
            if (p2[1] > p1[1]) & (1.25 > gradient > 0.75) & (p2[1] > 500):
                print("Match Found")
                print(p1)
                print(p2)
                self.bottomLine = bottleContour[i][0]
                cv.circle(self.img, (self.bottomLine[0], self.bottomLine[1]), 1, (0, 0, 255))
                self.show()
                break

    def filterBlue(self):
        blank = np.zeros(self.img.shape[:2], dtype='uint8')
        for x in range(self.img.shape[0]):
            for y in range(self.img.shape[1]):
                pixel = self.img[x][y]
                if x == 400:
                    pdb.set_trace()
                if float(pixel[0]) > float(pixel[1] + pixel[2]) * 0.7:
                    blank[x][y] = 0
                else:
                    blank[x][y] = 255

        cv.imshow("Mask", blank)
        cv.waitKey(0)

    def backgroundRemoval(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(self.img, 150, 255, cv.THRESH_BINARY)
        self.img = self.original
        self.img = cv2.bitwise_and(self.img, self.img, mask=mask)
        # self.img = cv.morphologyEx(self.img, cv.MORPH_OPEN, (3, 3), iterations=1)

    def twinBoxAlgorithm(self):
        self.backgroundRemoval()
        self.show()

        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.blur(5)
        self.show()

        self.edge()
        self.getContour()
        self.scanForBottomCorner()
        cv2.line(self.img, (100, self.bottomLine[1]),
                 (200, self.bottomLine[1]), (255, 255, 255), 3)
        self.show()

        self.findContours()
        self.getBoundingBox()
        self.drawAllContours()
        self.show()

    def experiment(self, num):
        if num == 1:
            self.radius = 75 / 2
            # self.original = cv.imread("data/EX14.jpg")
            self.original = cv.imread("data/EX21.jpg")
            self.img = self.original
            self.resizeAspect(height=900)
            self.show()

            # PROCESSING PIPELINE
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            self.blur(7)
            self.show()

            self.edge()
            self.img = cv.dilate(self.img, (5, 5))
            self.img = cv.erode(self.img, (5, 5))
            self.show()

            self.findContours()
            self.getBoundingBox()
            self.drawAllContours()
            self.show()

            # ANALYSES PIPELINE
            self.img = self.original
            waterLine = self.boundingBox[1][0:2]
            cv2.line(self.img, (int(waterLine[0][0] - 20), int(waterLine[0][1])),
                     (int(waterLine[1][0] + 20), int(waterLine[1][1])), [10, 10, 255], 3)
            self.scanForBottomCorner()
            cv2.line(self.img, (int(waterLine[0][0] - 20), self.bottomLine[1]),
                     (int(waterLine[1][0] + 20), self.bottomLine[1]), (0, 0, 255), 3)
            self.show()

            pixelPerMetric = (self.boundingBox[0][1][0] - self.boundingBox[0][0][0]) / 84
            height = (self.bottomLine[1] - waterLine[0][1]) / pixelPerMetric
            volume = math.pi * (75 * 75 / 4) * height / 1000

            cv2.putText(self.img, "{:.1f}mL".format(volume), (int(waterLine[1][0] + 30), int(waterLine[1][1])),
                        cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            self.show()
        if num == 2:
            self.radius = 75 / 2
            self.original = cv.imread("data/EX16.jpg")
            self.img = self.original
            self.resizeAspect(height=900)
            self.show()

            # PROCESSING PIPELINE
            self.twinBoxAlgorithm()

            # ANALYSES PIPELINE
            self.img = self.original
            waterLine = self.boundingBox[1][0:2]
            cv2.line(self.img, (int(waterLine[0][0] - 20), int(waterLine[0][1])),
                     (int(waterLine[1][0] + 20), int(waterLine[1][1])), [10, 10, 255], 3)
            self.scanForBottomCorner()
            cv2.line(self.img, (int(waterLine[0][0] - 20), self.bottomLine[1]),
                     (int(waterLine[1][0] + 20), self.bottomLine[1]), (0, 0, 255), 3)
            self.show()

            pixelPerMetric = (self.boundingBox[0][1][0] - self.boundingBox[0][0][0]) / 84
            height = (waterLine[1][0] - waterLine[0][0]) / pixelPerMetric
            volume = math.pi * (75 * 75 / 4) * height / 1000

            cv2.putText(self.img, "{:.1f}mL".format(volume), (int(waterLine[1][0] + 30), int(waterLine[1][1])),
                        cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            self.show()

            pdb.set_trace()


def main():
    print("Loading camera")
    capture = cv.VideoCapture(3)
    print("Camera ready")
    ret, frame = capture.read()
    image = Vision(img=frame)
    while True:

        ret, frame = capture.read()
        image.updateImage(frame)
        image.gray()
        image.blur()
        image.edge()
        image.show(False)

        if cv.waitKey(1) == 27:
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    image = Vision()
    image.experiment(1)
    cv.destroyAllWindows()
