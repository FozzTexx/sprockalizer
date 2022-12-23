import math
import numpy as np

def isarray(a):
  return isinstance(a, list) or isinstance(a, tuple) or isinstance(a, np.ndarray)

class Point:
  def __init__(self, point, y=None):
    if isinstance(point, list) or isinstance(point, tuple):
      self.point = point
    elif isinstance(point, np.ndarray):
      self.point = point.tolist()
    elif y is not None:
      self.point = (point, y)
    else:
      raise ValueError("Passed arguments do not make a coordinate")
    return

  @property
  def x(self):
    return self.point[0]

  @x.setter
  def x(self, val):
    self.point = (val, self.point[1])
    return
  
  @property
  def y(self):
    return self.point[1]

  @y.setter
  def y(self, val):
    self.point = (self.point[0], val)
    return

  @property
  def cv(self):
    return tuple(np.array(self.point).astype(int))

  def midpoint(self, other):
    return Point((self.x + (other.x - self.x) / 2, self.y + (other.y - self.y) / 2))

  def __repr__(self):
    return "%s: (%0.3f, %0.3f)" % (self.__class__.__name__, *self.point[:2])
  
class Size:
  def __init__(self, size, height=None):
    if isinstance(size, list) or isinstance(size, tuple):
      self.size = size
    elif isinstance(size, np.ndarray):
      self.size = size.tolist()
    elif height is not None:
      self.size = (size, height)
    else:
      raise ValueError("Passed arguments do not make a size")
    return

  @property
  def width(self):
    return self.size[0]

  @width.setter
  def width(self, val):
    self.size = (val, self.size[1])
    return
  
  @property
  def height(self):
    return self.size[1]

  @height.setter
  def height(self, val):
    self.size = (self.size[0], val)
    return

  @property
  def aslist(self):
    return [self.width, self.height]
  
  def __repr__(self):
    return "%s: (%0.3f, %0.3f)" % (self.__class__.__name__, *self.size[:2])
  
class Line:
  def __init__(self, x1, y1=None, x2=None, y2=None):
    if isarray(x1):
      if len(x1) == 4:
        self.start = Point(x1[:2])
        self.end = Point(x1[2:])
      elif len(x1) == 2 and isarray(y1):
        self.start = Point(x1)
        self.end = Point(y1)
    else:
      self.start = Point(x1, y1)
      self.end = Point(x2, y2)
    return

  def intersection(self, aLine):
    xdiff = (self.x1 - self.x2, aLine.x1 - aLine.x2)
    ydiff = (self.y1 - self.y2, aLine.y1 - aLine.y2)

    def det(a, b):
      return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
      return None

    d = (det(self.start.point, self.end.point),
         det(aLine.start.point, aLine.end.point))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return Point((x, y))

  @property
  def x1(self):
    return self.start.x

  @property
  def x2(self):
    return self.end.x

  @property
  def y1(self):
    return self.start.y

  @property
  def y2(self):
    return self.end.y

  @property
  def angle(self):
    xdiff = self.start.x - self.end.x
    ydiff = self.start.y - self.end.y
    angle = 360 - math.degrees(math.atan2(xdiff, ydiff)) - 90
    if angle > 360:
      angle -= 360
    elif angle < 0:
      angle += 360
    return angle

  @property
  def length(self):
    return math.sqrt((self.start.x - self.end.x) ** 2 \
                     + (self.start.y - self.end.y) ** 2)

class Bounds:
  def __init__(self, origin, size=None, br=None):
    if isarray(origin) and len(origin) == 4:
        self.origin = Point(origin[:2])
        self.size = Size(origin[2:])
    else:
      if isarray(origin):
        self.origin = Point(origin)
      else:
        self.origin = origin
      if size is not None:
        if isarray(size):
          self.size = Size(size)
        else:
          self.size = size
      elif br is not None:
        if isarray(br):
          br = Point(br)
        self.size = Size(br.x - self.origin.x, br.y - self.origin.y)
    return

  @property
  def x(self):
    return self.origin.x

  @property
  def y(self):
    return self.origin.y

  @property
  def width(self):
    return self.size.width

  @property
  def height(self):
    return self.size.height

  @property
  def x1(self):
    return self.x

  @x1.setter
  def x1(self, val):
    self.origin.x = val
    return
  
  @property
  def y1(self):
    return self.y

  @y1.setter
  def y1(self, val):
    self.origin.y = val
    return
  
  @property
  def x2(self):
    return self.origin.x + self.size.width

  @x2.setter
  def x2(self, val):
    self.size.width = self.origin.x + val
    return
  
  @property
  def y2(self):
    return self.origin.y + self.size.height

  @y2.setter
  def y2(self, val):
    self.size.height = self.origin.y + val
    return
  
  @property
  def topLeft(self):
    return self.origin

  @property
  def topRight(self):
    return Point(self.x2, self.y)

  @property
  def bottomLeft(self):
    return Point(self.x, self.y2)

  @property
  def bottomRight(self):
    return Point(self.x2, self.y2)

  @property
  def area(self):
    return self.size.width * self.size.height

  @property
  def cv(self):
    return [self.topLeft.cv, self.bottomRight.cv]

  @property
  def aslist(self):
    return [self.x1, self.y1, self.width, self.height]
  
  def intersection(self, aRect):
    x1 = max(self.x1, aRect.x1)
    y1 = max(self.y1, aRect.y1)
    x2 = min(self.x2, aRect.x2)
    y2 = min(self.y2, aRect.y2)
    if x1 <= x2 and y1 <= y2:
      return Rectangle(origin=(x1, y1), br=(x2, y2))
    return None

  def __repr__(self):
    return "%s: ((%i, %i), (%i, %i))" % (self.__class__.__name__,
                                         self.x1, self.y1, self.width, self.height)

def line_angle(line):
  return math.atan2(line[0][1] - line[1][1], line[0][0] - line[1][0])

def linify(lines, img, minlen):
  vertical = []
  horizontal = []
  thickness = 2
  ANGLE_LIMIT = 3

  if lines is not None:
    # print("LINIFY", len(lines), lines.shape, lines.ndim)
    if lines.ndim != 3:
      for p in lines:
        h, v = linify(p, img, minlen)
        horizontal.extend(h)
        vertical.extend(v)
    else:
      for x in range(0, len(lines)):
        for le in lines[x]:
          if len(le) == 4:
            x1,y1,x2,y2 = le
          elif len(le) == 2:
            x1,y1 = le
            nx = (x+1) % len(lines)
            x2,y2 = lines[nx][0]
          line = ((x1, y1), (x2, y2))
          l = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
          if l < minlen:
            continue
          angle = abs(math.degrees(line_angle(line)))
          if abs(180 - angle) <= ANGLE_LIMIT:
            horizontal.append(line)
          elif abs(90 - angle) <= ANGLE_LIMIT and l > 10:
            vertical.append(line)

    if img is not None:
      draw_lines(img, vertical, CLR_LGREEN, thickness)
      draw_lines(img, horizontal, CLR_LRED, thickness)

  return horizontal, vertical

  
