import cv2


def define_points() :
    balls = [(100,100), (120, 100), (140,100)]

    return balls




def __main__():
    points = define_points()

    sprite = cv2.imread("./sprite.webp")

    for point in points:
        cv2.circle(sprite, point, 5, (0,20,220), -1)
    
    cv2.imshow("Sprite", sprite)
    cv2.waitKey(0)


__main__()
