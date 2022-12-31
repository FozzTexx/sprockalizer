from line import Bounds
import cv2
from scipy import stats
from scipy.signal import find_peaks
import numpy as np
import math
from color import *

def get_mode(arr, col):
  if col is not None:
    a = arr[:, col]
  else:
    a = arr
  a = a[a[:] != None]
  if len(a) == 0:
    return None
  return stats.mode(a)[0][0]

class Gate:
  MARGIN = 20

  def __init__(self, frame, sprocket):
    self.bounds = None

    if frame is None:
      return

    cv2.namedWindow("3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("4", cv2.WINDOW_NORMAL)

    original = frame.copy()
    gray = frame
    if len(gray.shape) == 3:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    gblack = gray[int(h / 2), w - 1]
    mblack = np.where(gray <= gblack + 10)
    mwhite = np.where(gray > gblack + 10)
    gray[mblack] = 0
    gray[mwhite] = 255
    cols = np.sum(gray, axis=0)
    rows = np.sum(gray, axis=1)
    g_rows = 1 - (rows / (w * 255))
    g_cols = 1 - (cols / (h * 255))

    est_top = est_bot = gate_height = None
    if sprocket.top and sprocket.bottom:
      v_center = int(sprocket.height / 2 + sprocket.top)
      gate_height = sprocket.estimatedFrameBounds.height
      est_top = v_center - gate_height / 2
      est_bot = est_top + gate_height
      if est_bot >= frame.shape[0]:
        est_bot = frame.shape[0] - 1
        est_top = est_bot - gate_height
      t_offset = max(int(est_top - self.MARGIN), 0)
      b_offset = int(est_bot - self.MARGIN)

      top_rows = g_rows[t_offset:t_offset + self.MARGIN * 2]
      bot_rows = g_rows[b_offset:b_offset + self.MARGIN * 2]

      cv2.line(frame, (0, int(est_top)), (frame.shape[1], int(est_top)), CLR_LBLUE, 5)
      cv2.line(frame, (0, int(est_top + gate_height)),
               (frame.shape[1], int(est_top + gate_height)), CLR_LBLUE, 5)

      print("ESTIMATED",
            "sprock:", sprocket.height,
            "mm:", sprocket.mm_scale,
            "gate:", gate_height,
            "top:", est_top)
    else:
      t_offset = 0
      b_offset = v_center = int(h / 2)
      top_rows = g_rows[:v_center]
      bot_rows = g_rows[v_center:]

    MARGIN = 0.02
    # print("TOP ROWS", len(top_rows))
    # print("BOT ROWS", b_offset, len(bot_rows))
    tpeaks = self.get_peaks(top_rows, MARGIN)
    bpeaks = self.get_peaks(bot_rows, MARGIN)
    used_bounds = used_border = False

    tpeaks += t_offset
    bpeaks += b_offset

    if est_top:
      # print("TPEAKS", tpeaks)
      # print("BPEAKS", bpeaks)
      # if not len(bpeaks):
      #   print("BROWS", bot_rows)
      est_top = self.best_row(tpeaks, est_top, 40)
      est_bot = self.best_row(bpeaks, est_bot, 40)
      # print("ESTIMATED", est_top, est_bot)
    elif 0 < len(tpeaks) <= 5 and 0 < len(bpeaks) <= 5:
      est_top = min(tpeaks)
      est_bot = max(bpeaks)

    x1 = sprocket.right
    y1 = est_top
    y2 = est_bot
    print("GATE GUESS", x1, y1, y2)
    if x1 is not None and y1 is not None and y2 is not None:
      x2 = int(x1 + sprocket.estimatedFrameBounds.width)
      if x2 >= frame.shape[1]:
        x2 = frame.shape[1] - 1
        
      r_offset = x2 - 20
      rgt_cols = g_cols[r_offset:]
      cmax = np.max(rgt_cols)
      cmax2 = cmax - MARGIN
      w = np.where(rgt_cols < cmax - MARGIN)
      if len(w[0]):
        cmax2 = np.max(rgt_cols[w])
      rmax = np.max(g_cols) - 0.15
      rpeak = np.where(rgt_cols >= rmax)[0]
      rpeak += r_offset
      rdiff = np.diff(rpeak)
      x2 = self.best_row(rpeak, x2, 100)
      self.bounds = Bounds((int(x1), int(y1)), br=(int(x2), int(y2)))
      # print("GATE", self.bounds)

      # cv2.line(frame, (0, int(y1)), (frame.shape[1], int(y1)), CLR_GREEN, 5)
      # cv2.line(frame, (0, int(y2)), (frame.shape[1], int(y2)), CLR_GREEN, 5)
      # cv2.line(frame, (x1, 0), (x1, frame.shape[0]), CLR_GREEN, 5)
      # cv2.line(frame, (x2, 0), (x2, frame.shape[0]), CLR_GREEN, 5)
    else:
      print("TOO MANY PEAKS", len(tpeaks), len(bpeaks))
      print(tpeaks)
      print(bpeaks)

    if True or self.bounds:
      top, bottom, right = self.blobify(original, sprocket)
      print("BLOB", top, bottom, right)

      MARGIN = 20
      PEAK_MARGIN = 0.1
      est_top = self.estimate(top, MARGIN, PEAK_MARGIN, g_rows)
      est_bot = self.estimate(bottom, MARGIN, PEAK_MARGIN, g_rows)
      est_rgt = self.estimate(right, MARGIN, PEAK_MARGIN, g_cols)
      
      if (abs(top - self.bounds.y1) < MARGIN * 4 or est_top) \
         and (abs(bottom - self.bounds.y2) < MARGIN * 4 or est_bot):
        if abs(right - self.bounds.x2) > MARGIN * 4 and est_rgt is None:
          print("NO RIGHT", right, self.bounds.x2, abs(right - self.bounds.x2))
          right = self.bounds.x2
        print("BLOB2", top, bottom, right)
        # FIXME - do something about left/x1?
        self.bounds = Bounds((self.bounds.x1, top), br=(right, bottom))
        print("NEW BOUNDS", self.bounds.topLeft, self.bounds.bottomRight)
      else:
        print("BLOB FAIL", abs(top - self.bounds.y1), abs(bottom - self.bounds.y2))

    lcheck = np.ones(shape=frame.shape, dtype=np.uint8)
    lcheck *= 255
    for y, v in enumerate(top_rows):
      x = int(v * lcheck.shape[1])
      cv2.line(lcheck, (0, y + t_offset), (x, y + t_offset), 0, 5)
    for y, v in enumerate(bot_rows):
      x = int(v * lcheck.shape[1])
      cv2.line(lcheck, (0, y + b_offset), (x, y + b_offset), 0, 5)
    # if est_top and est_bot:
    #   cv2.line(lcheck, (0, int(est_top)), (lcheck.shape[1], int(est_top)), CLR_LBLUE, 5)
    #   cv2.line(lcheck, (0, int(est_bot)), (lcheck.shape[1], int(est_bot)), CLR_LBLUE, 5)

    cv2.imshow("3", lcheck)

    lcheck = np.ones(shape=frame.shape, dtype=np.uint8)
    lcheck *= 255
    for x, v in enumerate(g_cols):
      y = int(v * lcheck.shape[0])
      cv2.line(lcheck, (x, 0), (x, y), 0, 5)
    cv2.imshow("4", lcheck)

    return

  def blobify(self, frame, sprocket):
    gray = frame
    if len(gray.shape) == 3:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bottom = -(sprocket.estimatedFrameBounds.y2 - 75)
    top = -(sprocket.estimatedFrameBounds.y1 + 75)
    right = -(sprocket.estimatedFrameBounds.x2 - 200)
    alt_top = alt_bottom = None
    STEPS = 8
    MARGIN = 20
    for step in range(STEPS - 2):
      thresh = 255 / STEPS * (step + 1)
      print("THRESHOLD", thresh)
      _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
      binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
      cv2.line(binary_bgr, (0, self.bounds.y1), (binary_bgr.shape[1], self.bounds.y1),
               CLR_YELLOW, 5)
      cv2.line(binary_bgr, (0, self.bounds.y2), (binary_bgr.shape[1], self.bounds.y2),
               CLR_YELLOW, 5)
      
      contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for c in contours:
        rect = Bounds(cv2.boundingRect(c))
        if rect.width < 100 or rect.height < 100:
          continue
        useRect = False
        if rect.y1 >= self.bounds.y1 - MARGIN and rect.y2 <= self.bounds.y2 + MARGIN \
           and rect.y2 > self.bounds.y1 + MARGIN * 4 and rect.x2 < frame.shape[1] - 15:
          print("GOOD RECT", frame.shape[1], rect.x2, self.bounds, rect)
          if rect.y1 < abs(top):
            top = rect.y1
          if rect.y2 > abs(bottom):
            bottom = rect.y2
          r_edge = self.bounds.x1 + sprocket.estimatedFrameBounds.width
          if rect.x2 > abs(right) and abs(rect.x2 - r_edge) < MARGIN * 4:
            print("GOOD RIGHT", rect.x2, self.bounds.x2)
            right = rect.x2
          else:
            print("BAD RIGHT", right, rect.x2, r_edge, self.bounds.x2,
                  rect.x2 > abs(right), abs(rect.x2 - r_edge) < MARGIN * 4)
          cv2.rectangle(binary_bgr, *rect.cv, CLR_RED, 5)
        elif rect.y1 >= self.bounds.y2 - MARGIN and rect.y2 == frame.shape[0]:
          if alt_bottom is None or rect.y1 < alt_bottom:
            alt_bottom = rect.y1
          print("BOTTOM FRAME", alt_bottom, rect)  
          cv2.rectangle(binary_bgr, *rect.cv, CLR_GREEN, 5)
        elif rect.y1 == 0 and rect.y2 <= self.bounds.y1 + MARGIN:
          if alt_top is None or rect.y2 > alt_top:
            alt_top = rect.y2
          print("TOP FRAME", alt_top, rect)  
          cv2.rectangle(binary_bgr, *rect.cv, CLR_GREEN, 5)
        else:
          print("BAD RECT", top, bottom, rect)
          print(rect.y1 >= self.bounds.y1 - MARGIN,
                rect.y1 < abs(top),
                rect.y2 <= self.bounds.y2 + MARGIN,
                rect.y2 > abs(bottom))
          cv2.rectangle(binary_bgr, *rect.cv, CLR_BLUE, 5)

      #cv2.drawContours(binary, contours, -1, 127, 5)
      cv2.namedWindow("blob", cv2.WINDOW_NORMAL)
      cv2.imshow("blob", binary_bgr)
      cv2.waitKey(0)

    if top < 0:
      top = self.bounds.y1
    if bottom < 0:
      bottom = self.bounds.y2
    if right < 0:
      right = self.bounds.x2
    print("BLOB TOP BOT", top, bottom, right, alt_bottom, alt_top,
          abs(top - self.bounds.y1), abs(bottom - self.bounds.y2),
          abs(alt_top - self.bounds.y1) if alt_top is not None else None,
          abs(alt_bottom - self.bounds.y2) if alt_bottom is not None else None)
    if alt_top is not None and alt_top > top:
      top = alt_top
    if alt_bottom is not None and alt_bottom > bottom:
      bottom = alt_bottom

    # if abs(top - self.bounds.y1) < MARGIN * 4 \
    #    and abs(bottom - self.bounds.y2) < MARGIN * 4:
    #   if abs(right - self.bounds.x2) > MARGIN * 4:
    #     print("NO RIGHT", right, self.bounds.x2, abs(right - self.bounds.x2))
    #     right = self.bounds.x2
    #   # FIXME - do something about left/x1?
    #   self.bounds = Bounds((self.bounds.x1, top), br=(right, bottom))
    #   print("NEW BOUNDS", self.bounds)
    # else:
    #   print("BLOB FAIL", abs(top - self.bounds.y1), abs(bottom - self.bounds.y2))
    return top, bottom, right

  @staticmethod
  def best_row(rows, val, margin, canFail=False):
    if len(rows):
      idx = (np.abs(rows - val)).argmin()
      best = rows[idx]
      # print("BEST", best)
      if abs(best - val) <= margin:
        val = best
      elif canFail:
        val = None
    return val

  @staticmethod
  def get_peaks(rows, margin):
    v_max = np.max(rows)
    v_max2 = v_max - margin
    w = np.where(rows < v_max - margin)
    if len(w[0]):
      v_max2 = np.max(rows[w])
    peaks, _ = find_peaks(rows, v_max2 - margin)
    if not len(peaks):
      peaks = np.where(rows == v_max)[0]
    return peaks

  def estimate(self, position, margin, peak_margin, rows):
    est = None
    offset = int(position - margin)
    section = rows[offset:offset + margin * 2]
    if len(section):
      peaks = self.get_peaks(section, peak_margin)
      if len(peaks):
        peaks += offset
        est = self.best_row(peaks, position, margin / 2, canFail=True)
    return est
