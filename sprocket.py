import cv2
import numpy as np
from dataclasses import dataclass
from itertools import zip_longest
from line import Size, Bounds, linify
from color import *

# Dimensions in mm
SPROCKET_SUPER8 = (0.914, 1.09)
FRAME_SUPER8 = (5.80, 4.10)
GATE_SUPER8 = (5.79, 4.01)
SPROCKET_8MM = (1.76, 1.22, 2.50)
FRAME_8MM = (5.42, 3.76)
GATE_8MM = (4.5, 3.3)

@dataclass
class Dimensions:
  sprocket: Size
  frame: Size
  gate: Size
  holes: int

class Sprocket:
  def __init__(self, frame, mode=None):
    self.top = self.right = self.bottom = None
    if mode == "8mm":
      # Regular 8 should have 2 sprockets near the top & bottom
      self.standard = Dimensions(sprocket=Size(SPROCKET_8MM[0], SPROCKET_8MM[2]),
                                 frame=Size(FRAME_8MM),
                                 gate=Size(GATE_8MM),
                                 holes=2)
    else:
      # Super 8 sprocket should be near the center
      self.standard = Dimensions(sprocket=Size(SPROCKET_SUPER8),
                                 frame=Size(FRAME_SUPER8),
                                 gate=Size(GATE_SUPER8),
                                 holes=1)

    if frame is not None:
      self.resolution = Size(frame.shape[1::-1])

      frame_f = frame.copy()
      mask = np.zeros((self.resolution.height + 2, self.resolution.width + 2), np.uint8)
      dist = (3,)
      if len(frame.shape) == 3:
        dist = dist * frame.shape[2]

      holes = self.sprocket_guess(frame)
      if len(holes) != self.standard.holes:
        v_center = self.resolution.height / 2
        above = []
        below = []
        for h in holes:
          if h.y1 < v_center:
            above.append([abs(v_center - (h.y1 + h.height / 2)), h.height, h])
          else:
            below.append([abs(v_center - (h.y1 + h.height / 2)), h.height, h])
        above.sort(key=lambda x: x[0])
        below.sort(key=lambda x: x[0])
        if self.standard.holes > 1:
          combined = np.array([x[1] for x in above + below])
          avg = np.mean(combined)
          above = [x for x in above if x[1] - avg > -10]
          below = [x for x in below if x[1] - avg > -10]
        
        holes = [ele[2] for comb in zip_longest(above, below)
                 for ele in comb if ele is not None]
        holes = holes[:self.standard.holes]

      for hole in holes:
        pos = int(hole.y1 + hole.height / 2)
        cv2.floodFill(frame_f, mask, (0, pos), 255, loDiff=dist, upDiff=dist)

      sprocket_c = mask[1:-1, 1:-1] * 255
      contours, _ = cv2.findContours(sprocket_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      lcheck = np.stack((sprocket_c,)*3, axis=-1)
      cv2.drawContours(lcheck, contours, -1, CLR_GREEN, 5)
      sprocket_c = cv2.Canny(sprocket_c, 50, 100, apertureSize=3)

      lines1 = cv2.HoughLinesP(sprocket_c, 2, np.pi/180, threshold=10,
                               minLineLength=5, maxLineGap=5)
      spr_h, spr_v = linify(lines1, None, 5)
      for line in spr_h + spr_v:
        cv2.line(lcheck, *line, CLR_RED, 5)

      if spr_h is not None and spr_v is not None:
        self.find_sprocket(self.resolution, np.array(spr_h), np.array(spr_v),
                           np.array(contours), mode)

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

  def find_sprocket(self, size, horizontal, vertical, contours, mode):
    MARGIN = int(20 / 1080 * size.width)
    RIGHT = int(240 / 1440 * size.width)
    CMAX = int(200 / 1080 * size.height)
    v_center = size.height / 2

    horz = horizontal.reshape(-1, 2)
    vert = vertical.reshape(-1, 2)
    cont = []
    rmin = None
    if len(vert):
      rmin, _, _ = self.discard_outliers(vert[:, 0])

    for c in contours:
      rect = Bounds(cv2.boundingRect(c))
      area = cv2.contourArea(c)
      if rmin is not None and rect.width < rmin:
        continue
      cont.extend(c.reshape(-1, 2))
    cont = np.array(cont)

    w = np.where(vert[:, 0] < RIGHT)
    pts_x = vert[w][:, 0]
    if len(pts_x):
      _, _, remain = self.discard_outliers(pts_x)
      if len(remain):
        self.right = int(np.max(remain))
    if self.right is None and len(cont):
      w = np.where(cont[:, 0] < RIGHT)
      pts_x = cont[w][:, 0]
      if len(pts_x):
        _, _, remain = self.discard_outliers(pts_x)
        if len(remain):
          self.right = int(np.max(remain))

    w = np.where((horz[:, 1] < v_center) & (horz[:, 0] < RIGHT - MARGIN))
    pts_y = horz[w][:, 1]
    if len(pts_y):
      _, _, remain = self.discard_outliers(pts_y)
      if len(remain):
        self.top = int(np.max(remain))
    if self.top is None and len(cont):
      w = np.where((cont[:, 1] < v_center) & (cont[:, 0] < RIGHT - MARGIN))
      pts_y = cont[w][:, 1]
      if len(pts_y):
        _, _, remain = self.discard_outliers(pts_y)
        if len(remain):
          if mode == "8mm":
            self.top = int(np.max(remain))
          else:
            self.top = int(np.min(remain))

    w = np.where((horz[:, 1] > v_center) & (horz[:, 0] < RIGHT - MARGIN))
    pts_y = horz[w][:, 1]
    if len(pts_y):
      _, _, remain = self.discard_outliers(pts_y)
      if len(remain):
        self.bottom = int(np.min(remain))
    if self.bottom is None and len(cont):
      w = np.where((cont[:, 1] > v_center) & (cont[:, 0] < RIGHT - MARGIN))
      pts_y = cont[w][:, 1]
      if len(pts_y):
        _, _, remain = self.discard_outliers(pts_y)
        if len(remain):
          if mode == "8mm":
            self.bottom = int(np.min(remain))
          else:
            self.bottom = int(np.max(remain))

    return

  def sprocket_guess(self, frame):
    # Look for sprockets in leftmost 10 columns
    cols = frame[:, :10]
    if len(cols.shape) > 2:
      cols = cv2.cvtColor(cols, cv2.COLOR_BGR2GRAY)
    cols = cv2.GaussianBlur(cols, (5, 5), 0)
    # Find brightest pixel
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cols)
    _, binary = cv2.threshold(cols, maxVal - 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
      rects.append(Bounds(cv2.boundingRect(c)))
    return rects

  def discard_outliers(self, array):
    q75, q25 = np.percentile(array, [75, 25])
    intr_qr = q75 - q25
    qmax = q75 + (1.5 * intr_qr)
    qmin = q25 - (1.5 * intr_qr)
    remain = np.array(array)
    w = np.where(remain >= qmin)
    return qmin, qmax, remain[w]

  @property
  def found(self):
    return self.top is not None and self.right is not None and self.bottom is not None

  @property
  def bounds(self):
    if self.found:
      return Bounds((0, self.top), br=(self.right, self.bottom))
    return None

  @property
  def height(self):
    if self.top is not None and self.bottom is not None:
      return self.bottom - self.top
    return None

  @property
  def mm_scale(self):
    if self.top and self.bottom:
      return self.height / self.standard.sprocket.height
    return None

  @property
  def gateBounds(self):
    bounds = self.bounds
    if bounds:
      x1 = bounds.x2
      mm_scale = bounds.height / self.standard.sprocket.height
      v_center = int(bounds.height / 2 + bounds.y1)
      gate_height = int(self.gateSize.height * mm_scale)
      y1 = int(v_center - gate_height / 2)
      y2 = y1 + gate_height
      aspect = self.gateSize.width / self.gateSize.height
      x2 = int(x1 + (y2 - y1) * aspect)
      return Bounds((x1, y1), br=(x2, y2))
    return None

  @property
  def guess(self):
    if not self.found:
      return None
    scale = self.height / self.standard.sprocket.height
    center = (self.top + self.bottom) / 2
    left = self.right
    top = int(center - gateSize.height * scale / 2)
    right = left + int(gateSize.width * scale)
    bottom = top + int(gateSize.height * scale)
    if top >= 0 and bottom < self.resolution.height and right < self.resolution.width:
      return Bounds((left, top), br=(right, bottom))
    return None

  @property
  def stats(self):
    return [self.top, self.right, self.bottom, self.height]
