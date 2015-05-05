import cv2
import numpy as np
import math
import time
import itertools
from port import Port

N = 1
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

COLOR = [BLUE]

P_WIDTH = 400
PERIOD = 5


def get_robot1(frame):
    lower = np.array([67,95,93])
    upper = np.array([102,255,255])
    return get_pos(frame, lower, upper, 1)

def get_robot2(frame):
    lower = np.array([59, 6, 194])
    upper = np.array([255,255, 255])
    return get_pos(frame, lower, upper, 1)

def get_white(frame):
    lower = np.array([235,235,235])
    upper = np.array([255,255,255])
    return get_pos(frame, lower, upper, N, True)


def get_pos(frame, lower_bound, upper_bound, n, isRGB = False):
    distance = 500 if n == 1 else 50
    hsv = frame if isRGB else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask= mask)
    gray = cv2.cvtColor(res , cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT,1,distance, param1=50,param2=10,minRadius=5,maxRadius=25)
    if circles is None:
        print "no circle detection"
        return
    circles = np.round(circles[0, :]).astype("int")
    if len(circles) != n:
        print "number of circle %d != %d(we want)" % (len(circles), n)
        return


    """
    for (x, y, r) in circles:
        cv2.rectangle(frame, (x - 5, y    - 5), (x + 5, y + 5), (0, 128, 255), -1)
    cv2.imshow("gray", frame)
    """

    return circles

def ball_init(frame):
    lower_ball = np.array([26,29,164])
    upper_ball = np.array([180,255,255])
    #dp, minDist, param1, param2, minRadius, maxRadius = 1, 200, 50, 15, 15, 50

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    res = cv2.bitwise_and(frame, frame, mask= mask)
    gray = cv2.cvtColor(res , cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    #cv2.imshow("mask", mask)

    #circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, dp, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT,1,200, param1=50,param2=20,minRadius=10,maxRadius=50)

    #if circles is None or len(circles) != 4:
    if circles is None:
        return

    # convert the (x, y) coordinates and radius of  the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(frame, (x - 5, y    - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("output", frame)

def getbg():
    rval, bg = vc.read()
    while rval:
        rval, bg = vc.read()
        cv2.imshow("output", bg)
        # wait input
        key = cv2.waitKey(50) & 0xFF
        if key == 27: # exit on ESC
            return bg

def clickHandler(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONUP:
        return
    global mx, my
    x = int(x / 0.8)
    y = int(y / 0.8)
    if len(mx) == N:
        mx = [x]
        my = [y]
    else:
        mx.append(x)
        my.append(y)

def drawCircle(frame, pos, color):
    if pos is None:
        return
    for p in pos:
        if p is not None:
            r = p[2] * 2 if len(p) > 2 else 20
            cv2.circle(frame, (p[0], p[1]), r, color, -1)

def drawArrow(frame, fron, angle, color):
    headLen = 30
    bodyLen = 150
    to = (fron[0] + bodyLen * math.sin(math.radians(angle)), fron[1] - bodyLen * math.cos(math.radians(angle)))
    drawLine(frame, fron, to, color)
    drawLine(frame, to, (to[0] - headLen * math.cos(math.radians(angle) - math.pi/4), to[1] - headLen * math.sin(math.radians(angle) - math.pi/4)), color)

    drawLine(frame, to, (to[0] + headLen * math.cos(math.radians(angle) + math.pi/4), to[1] + headLen * math.sin(math.radians(angle) + math.pi/4)), color)

def drawLine(frame, x1, x2, color):
    if x1 is None or x2 is None:
        return
    cv2.line(frame, (int(x1[0]), int(x1[1])), (int(x2[0]), int(x2[1])), color, 10)

def getMovement(head, tail, dest):
    src = getMiddle(head, tail)
    dx = abs(int(dest[0] - src[0]))
    dy = abs(int(dest[1] - src[1]))

    carDir = getAngle(tail[0], tail[1], head[0], head[1])
    targetDir = getAngle(src[0], src[1], dest[0], dest[1])
    diffAngle = targetDir - carDir
    dist = distance(src, dest)
    print "dx = %d, dy = %d, distance = %d, %.2f %.2f %.2f" % (dx, dy, dist, carDir, targetDir, diffAngle)

    return dx, dy, diffAngle


def getAngle(x1, y1, x2, y2):
    angle = 0
    dx = float(x2 - x1)
    dy = float(y1 - y2)
    if dx > 0:
        angle = 90 - math.degrees(math.atan(dy/dx))
    elif dx < 0:
        angle = 270 - math.degrees(math.atan(dy/dx))
    else:
        angle = 0 if dy > 0 else 180
    return angle

def matchWhite(robots, whites):
    if whites is None:
        return whites
    for ws in itertools.permutations(whites):
        dis = []
        for i in range(0, N):
            if robots[i] is None:
                return
            dis.append(distance(robots[i], ws[i]))
        # check is tolerable
        tolerable = True
        for i in range(0, N):
            for j in range(i+1, N):
                if dis[i] > 200 or dis[j] > 200:
                    tolerable = False
                diff = float(dis[j]) / dis[i]
                if diff > 1.6 or diff < 0.4:
                    tolerable = False
        if tolerable:
            return ws
    return

def notNone(robots, whites):
    if robots is None or whites is None:
        return False
    if len(robots) != N or len(whites) != N:
        return False
    for i in robots:
        if i is None:
            return False
    for i in whites:
        if i is None:
            return False
    return True

def isInputValidate(robtos, whites):
    if not notNone(robtos, whites):
        return False
    return True



def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def drawMarker(frame, robots, whites):
    if not notNone(robots, whites):
        return
    for i in range(0, N):
        head = robots[i]
        tail = whites[i]
        middle = getMiddle(head, tail)
        color = COLOR[i]
        carDir = getAngle(tail[0], tail[1], head[0], head[1])

        # drawing
        drawCircle(frame, [middle], color)
        drawArrow(frame, middle, carDir, color)


def setUI(frame, robots, whites):
    # draw click circle
    global mx, my
    for i in range(0, len(mx)):
        cv2.circle(frame, (mx[i], my[i]), 10, WHITE, 3)

    # resize
    frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)

    height, width, depth = frame.shape
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, P_WIDTH, cv2.BORDER_CONSTANT, value=(250, 250, 250))

    if notNone(robots, whites):
        for i in range(0, N):
            middle = getMiddle(robots[i], whites[i])
            s = "%d    (x, y) = (%d, %d)" % (i, middle[0], middle[1])
            cv2.putText(frame, s, (width+30,50), cv2.FONT_HERSHEY_PLAIN, 1, BLACK, 2)


    cv2.imshow("output", frame)


def getMiddle(x, y):
    return ((x[0] + y[0]) / 2, (x[1] + y[1]) / 2)

mx = []
my = []

if __name__ == '__main__':
    cv2.namedWindow('output')
    cv2.setMouseCallback('output', clickHandler)
    port = Port()
    port.init()

    lasttimeSent = None

    prevRobot = [None] * N
    prevWhite = [None] * N

    vc = cv2.VideoCapture(1)
    balls = None
    rval, frame = vc.read()

    while rval:
        rval, frame = vc.read()
        robots = [None] * N
        whites = [None] * N

        origin = cv2.copyMakeBorder(frame, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

        # wait input
        key = cv2.waitKey(200) & 0xFF
        if key == 27: # exit on ESC
            break



        """
        # First ball should be initialized
        if balls == None and False:
            balls = ball_init(frame)
            if balls == None:
                print "ball is not initialized"
        else:
            print("do sth")
        """
        new_white = get_white(frame)
        drawCircle(frame, new_white, BLACK)


        r1 = get_robot1(frame)
        if (r1 is not None):
            robots[0] = r1[0]
        drawCircle(frame, [robots[0]], GREEN)

        """
        r2 = get_robot2(frame)
        if (r2 is not None):
            robtos[1] = r2[0]
        drawCircle(frame, [robtos[1]], BLUE)
        """

        if (new_white is not None):
            new_white = matchWhite(robots, new_white)
            if (new_white is not None):
                whites = new_white



        """
        for i in range(0, N):
            drawLine(frame, robtos[i], whites[i], BLUE)
        """


        if isInputValidate(robots, whites):
            prevWhite = whites
            prevRobot = robots
            if len(mx) == N:
                now = time.time()
                if lasttimeSent is None or now - lasttimeSent > PERIOD:
                    lasttimeSent = now
                    for i in range(0, N):
                        x, y, angle = getMovement(robots[i], whites[i], (mx[i], my[i]))
                        port.sendMessage(x, y, angle)

        #else:
        #    print "invalid"

        drawMarker(origin, prevRobot, prevWhite)

        setUI(origin, prevRobot, prevWhite)
        #setUI(frame)






    vc.release()
    cv2.destroyAllWindows()
