import line
import cv2
import numpy as np
from color import *

class Sprocket:
  # Dimensions in mm
  SPROCKET_SUPER8 = (0.914, 1.09)
  FRAME_SUPER8 = (5.77, 4.10)
  SPROCKET_8MM = ()
  FRAME_8MM = (4.5, 3.3)

  def __init__(self, frame):
    self.top = self.right = self.bottom = None

    if frame is not None:
      self.frameSize = line.Size(frame.shape[1::-1])

      v_center = int(self.frameSize.height / 2)
      frame_f = frame.copy()
      mask = np.zeros((self.frameSize.height + 2, self.frameSize.width + 2), np.uint8)
      dist = (2,)
      if len(frame.shape) == 3:
        dist = dist * frame.shape[2]
      cv2.floodFill(frame_f, mask, (0, v_center), 255, loDiff=dist, upDiff=dist)
      sprocket_c = mask[1:-1, 1:-1] * 255
      lcheck = frame & np.stack((sprocket_c,)*3, axis=-1)
      sprocket_c = cv2.Canny(sprocket_c, 50, 100, apertureSize=3)

      lines1 = cv2.HoughLinesP(sprocket_c, 2, np.pi/180, threshold=10,
                               minLineLength=5, maxLineGap=5)
      spr_h, spr_v = line.linify(lines1, None, 5)

      if spr_h is not None and spr_v is not None:
        self.find_sprocket(frame.shape[1::-1], spr_h, spr_v)

      if self.top:
        cv2.line(lcheck, (0, self.top), (frame.shape[1], self.top), CLR_YELLOW, 5)
      if self.bottom:
        cv2.line(lcheck, (0, self.bottom), (frame.shape[1], self.bottom),
                 CLR_YELLOW, 5)
      if self.right:
        cv2.line(lcheck, (self.right, 0), (self.right, frame.shape[0]),
                 CLR_YELLOW, 5)

      cv2.namedWindow("sprocket", cv2.WINDOW_NORMAL)
      cv2.imshow("sprocket", lcheck)
      
    return

  def find_sprocket(self, size, horizontal, vertical):
    MARGIN = int(20 / 1080 * size[0])
    RIGHT = int(140 / 1080 * size[1])
    CMAX = int(200 / 1080 * size[1])
    v_center = size[1] / 2

    pts_x = []
    pts_y = []
    for line in vertical:
      if ((line[0][1] > v_center - CMAX and line[0][1] < v_center + CMAX) \
          and (line[1][1] > v_center - CMAX and line[1][1] < v_center + CMAX)) \
         and line[0][0] < RIGHT and line[1][0] < RIGHT \
         and line[0][0] > 10 and line[1][0] > 10:
        pts_x.append(line[0][0])
        pts_x.append(line[1][0])
        pts_y.append(line[0][1])
        pts_y.append(line[1][1])
    if len(pts_x):
      counts = np.bincount(pts_x)
      x_min = np.argmax(counts)    
      edge = np.where(pts_x < x_min + MARGIN)
      RIGHT = self.right = int(np.median(np.array(pts_x)[edge]))

    pts_x = []
    pts_y = []
    for line in horizontal:
      if line[0][1] < v_center and line[1][1] < v_center \
         and line[0][0] < RIGHT - MARGIN and line[1][0] < RIGHT:
        pts_x.append(line[0][0])
        pts_x.append(line[1][0])
        pts_y.append(line[0][1])
        pts_y.append(line[1][1])
    if len(pts_x):
      y_max = np.max(pts_y)
      edge = np.where(pts_y > y_max - MARGIN)
      self.top = int(np.average(np.array(pts_y)[edge]))

    pts_x = []
    pts_y = []
    for line in horizontal:
      if line[0][1] > v_center and line[1][1] > v_center \
         and line[0][0] < RIGHT - MARGIN and line[1][0] < RIGHT:
        pts_x.append(line[0][0])
        pts_x.append(line[1][0])
        pts_y.append(line[0][1])
        pts_y.append(line[1][1])
    if len(pts_x):
      y_min = np.min(pts_y)
      edge = np.where(pts_y < y_min + MARGIN)
      self.bottom = int(np.average(np.array(pts_y)[edge]))
    return

  @property
  def found(self):
    return self.top is not None and self.right is not None and self.bottom is not None
  
  @property
  def bounds(self):
    if self.found:
      return line.Bounds((0, self.top), br=(self.right, self.bottom))
    return None

  @property
  def gateBounds(self):
    bounds = self.bounds
    if bounds:
      x1 = bounds.x2
      mm_scale = bounds.height / self.SPROCKET_SUPER8[1]
      v_center = int(bounds.height / 2 + bounds.y1)
      gate_height = int(self.FRAME_SUPER8[1] * mm_scale)
      y1 = int(v_center - gate_height / 2)
      y2 = y1 + gate_height
      aspect = self.FRAME_SUPER8[0] / self.FRAME_SUPER8[1]
      x2 = int(x1 + (y2 - y1) * aspect)
      return line.Bounds((x1, y1), br=(x2, y2))
    return None
      
  @property
  def guess(self):
    if not self.found:
      return None
    height = self.bottom - self.top
    scale = height / SPROCKET_SUPER8[1]
    center = (self.top + self.bottom) / 2
    left = self.right
    top = int(center - FRAME_SUPER8[1] * scale / 2)
    right = left + int(FRAME_SUPER8[0] * scale)
    bottom = top + int(FRAME_SUPER8[1] * scale)
    if top >= 0 and bottom < self.frameSize.height and right < self.frameSize.width:
      return line.Bounds((left, top), br=(right, bottom))
    return None

  @property
  def stats(self):
    height = None
    if self.top is not None and self.bottom is not None:
      height = self.bottom - self.top
    return [self.top, self.right, self.bottom, height]

