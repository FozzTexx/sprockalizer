import sprocket as sp
import line
import cv2
from scipy import stats
import numpy as np
import math

def find_contours(image):
  blank = np.zeros(shape=image.shape, dtype=np.uint8)
  contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  poly = []
  for cnt in contours:
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    poly.append(approx)
    cv2.drawContours(blank, [approx], -1, 255)
  kernel = np.ones((2,2), np.uint8) * 255
  blank = cv2.dilate(blank, kernel, iterations = 1)
  return blank, np.array(poly)

def longest_near(rc, dist, lines, xy):
  GROUPING = 3

  if not line.NN(lines):
    return None
  # FIXME - group lines into one score
  scores = []
  for li in lines:
    l = math.sqrt((li[0][0] - li[1][0]) ** 2 + (li[0][1] - li[1][1]) ** 2)
    r = (li[0][xy] + li[1][xy]) / 2
    d = abs(rc - r) + 1
    if d > dist:
      continue
    score = l / d
    scores.append([score, r, l, d])
  scores.sort(key=lambda x: x[1])
  scores = np.array(scores)
  #print("SCORES", scores)
  best = None
  idx = 0
  while idx < len(scores):
    score = scores[idx]
    close = scores[np.logical_and(scores[:, 1] >= score[1],
                                  scores[:, 1] <= score[1] + GROUPING)]
    #print("CLOSE", close)
    r = np.average(close[:, 1])
    l = np.sum(close[:, 2])
    d = np.average(close[:, 3])
    s = l / d
    #print("NEAR", rc, dist, s, r, l, d)
    if not line.NN(best) or s > best[0]:
      best = [s, r, l, d]
    idx += len(close)
  return best

def lines_near(lower, upper, lines, xy):
  pts = []
  if not line.NN(lines):
    return pts
  for li in lines:
    if li[0][xy] > lower and li[0][xy] < upper \
       and li[1][xy] > lower and li[1][xy] < upper:
      pts.append(li[0])
      pts.append(li[1])
  if not len(pts):
    return pts
  a = np.array(pts)
  return np.array(pts)[:, xy]

class Gate:
  def __init__(self, sprocket, frame, previous, spr_size):
    self.b_gate = self.l_gate = self.best = None
    
    gate_black = stats.mode(frame[:, -1])[0][0]
    gate = 255 * np.ones(shape=frame.shape, dtype=np.uint8)
    sel = np.where(frame <= gate_black + 30)
    gate[sel] = frame[sel]
    gate_t = gate.copy()
    gate_t[sel] = 0

    frame_f = gate.copy()
    h, w = frame.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(frame_f, mask, (w - 1, int(h / 2)), 255, loDiff=2, upDiff=2);
    gate = mask[1:-1, 1:-1] * 255

    gate_c, lines2 = find_contours(gate)
    frm_h, frm_v = line.linify(lines2, None, 20)

    ret, frame_t = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    gate_t, lines3 = find_contours(frame_t)
    frm2_h, frm2_v = line.linify(lines3, None, 20)

    if line.NN(frm2_h):
      frm_h += frm2_h
    if line.NN(frm2_v):
      frm_v += frm2_v    

    # draw_lines(lcheck, frm_h, CLR_LRED, 2)
    # draw_lines(lcheck, frm_v, CLR_LGREEN, 2)

    gate_i = 255 - gate
    newfr = self.find_gate(gate_i, previous, frm_h, frm_v, sprocket, spr_size)
    return

  def find_gate(self, frame, previous, horizontal, vertical, sprocket, spr_size):
    MARGIN = int(30 / 1080 * frame.shape[0])
    BLACK_LEVEL = 6 * frame.shape[0]

    self.b_gate = [None, None, None, None, None, None, None, None]
    guess = sprocket.guess
    if line.NN(guess):
      self.b_gate[0] = guess.x1
    self.l_gate = self.b_gate.copy()

    cols = np.sum(frame, axis=0)
    rows = np.sum(frame, axis=1)

    x = r = frame.shape[1]
    if line.NN(guess):
      x = guess.x2
    if x > r - MARGIN:
      x = r - MARGIN
    l = x - MARGIN
    re = np.where(cols[l:] < BLACK_LEVEL)
    if len(re[0]):
      self.b_gate[2] = np.min(re) + l
    else:
      rcols = cols[l:]
      print("RIGHT NOT FOUND", np.min(rcols))

    pts_x = lines_near(l, r, vertical, 0)
    if len(pts_x):
      self.l_gate[2] = int(np.median(pts_x))

    y = MARGIN
    if line.NN(previous):
      y = previous[0][1]
    elif line.NN(guess):
      y = guess.y1
    if y < MARGIN:
      y = MARGIN
    re = np.where(rows < BLACK_LEVEL)
    re2 = np.where(rows[y - MARGIN:y + MARGIN] < BLACK_LEVEL)
    if (len(re[0]) and np.max(re) < frame.shape[0] / 2) or len(re2[0]):
      if len(re2[0]):
        self.b_gate[1] = np.max(re2) + y - MARGIN
      else:
        self.b_gate[1] = np.max(re)
    else:
      trows = rows[:y + MARGIN]
      mrow = np.argmin(trows)
      if line.NN(previous) and abs(mrow - previous[0][1]) <= 5:
        self.b_gate[1] = mrow
      else:
        print("TOP NOT FOUND", np.argmin(trows), np.min(trows), y)

    row = longest_near(y, MARGIN, horizontal, 1)
    print("NEAREST-T", row)
    if line.NN(row):
      self.l_gate[1] = int(row[1])

    right = self.b_gate[2]
    top = self.b_gate[1]
    if not right:
      right = self.l_gate[2]
    if not top:
      top = self.l_gate[1]
    if not right or not top:
      print("ABORT")
      print(self.b_gate, "\n", self.l_gate, "\n", guess, "\n", previous)
      return None
    if line.NN(previous):
      height = previous[1][1] - previous[0][1]
    elif line.NN(guess):
      height = guess.height
    else:
      height = frame.shape[0] - MARGIN * 2 - top
    y = top + height
    if line.NN(guess):
      y = guess.y2
    if y > top + height:
      y = int(top + height)
    if y >= frame.shape[0]:
      y = frame.shape[0] - MARGIN - 1
    re = np.where(rows[y - MARGIN:] < BLACK_LEVEL)
    if len(re[0]):
      self.b_gate[3] = np.min(re) + y - MARGIN
    else:
      trows = rows[y - MARGIN:]
      mrow = np.argmin(trows) + y - MARGIN
      print("BOTTOM NOT FOUND", mrow, np.min(trows), y)

    row = longest_near(y, MARGIN, horizontal, 1)
    print("NEAREST-B", y, row)
    if line.NN(row):
      self.l_gate[3] = int(row[1])

    if line.NN(self.b_gate[2]) and line.NN(self.b_gate[0]):
      width = self.b_gate[2] - self.b_gate[0]
      height = (width / sp.FRAME_SUPER8[0]) * sp.FRAME_SUPER8[1]
      if line.NN(self.b_gate[1]):
        self.b_gate[7] = int(self.b_gate[1] + height)
      if line.NN(self.b_gate[3]):
        self.b_gate[5] = int(self.b_gate[3] - height)

    if line.NN(self.b_gate[3]) and line.NN(self.b_gate[1]):
      height = self.b_gate[3] - self.b_gate[1]
      width = (height / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
      if line.NN(self.b_gate[0]):
        self.b_gate[6] = int(self.b_gate[0] + width)
      if line.NN(self.b_gate[2]):
        self.b_gate[4] = int(self.b_gate[2] - width)

    if line.NN(self.l_gate[2]) and line.NN(self.l_gate[0]):
      width = self.l_gate[2] - self.l_gate[0]
      height = (width / sp.FRAME_SUPER8[0]) * sp.FRAME_SUPER8[1]
      if line.NN(self.l_gate[1]):
        self.l_gate[7] = int(self.l_gate[1] + height)
      if line.NN(self.l_gate[3]):
        self.l_gate[5] = int(self.l_gate[3] - height)

    if line.NN(self.l_gate[3]) and line.NN(self.l_gate[1]):
      height = self.l_gate[3] - self.l_gate[1]
      width = (height / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
      if line.NN(self.l_gate[0]):
        self.l_gate[6] = int(self.l_gate[0] + width)
      if line.NN(self.l_gate[2]):
        self.l_gate[4] = int(self.l_gate[2] - width)

    best = np.array(self.b_gate)
    missing = np.where(best == None)
    if len(missing[0]):
      best[missing] = np.array(self.l_gate)[missing]
    if None in best[:4]:
      print("TOO MUCH MISSING")
      print(best, "\n", self.b_gate, "\n", self.l_gate, "\n", guess, "\n", previous)
      return None

    if line.NN(best[1]) and line.NN(best[2]) \
       and line.NN(best[3]) and line.NN(best[7]):
      height = best[3] - best[1]
      w1 = best[2] - (height / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
      height = best[7] - best[1]
      w2 = best[2] - (height / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
      print("WIDTH", w1, w2)

    center = 0
    if sprocket.found and line.NN(best[1]) and line.NN(best[3]):
      g_center = guess.height / 2
      f_center = (best[1] + best[3]) / 2
      center = abs(f_center - g_center)
      print("CENTER", center)

    if None not in self.b_gate[:4] and None not in self.l_gate[:4] \
       and abs(self.b_gate[1] - self.l_gate[1]) < MARGIN / 2 \
       and abs(self.b_gate[2] - self.l_gate[2]) < MARGIN / 2 \
       and (abs(self.b_gate[3] - self.l_gate[3]) > MARGIN / 2 \
            or abs(best[3] - best[7]) > MARGIN / 2) \
       and (abs(self.b_gate[7] - self.l_gate[7]) > 5 or center > MARGIN / 2):
      print("BAD HEIGHT")
      best[3] = best[7]

    altw = alth = prvw = prvh = 0
    if line.NN(previous):
      prvw = previous[1][0] - previous[0][0]
      prvh = previous[1][1] - previous[0][1]
    curw = best[2] - best[0]
    curh = best[3] - best[1]
    aspw = (curh / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
    asph = (curw / sp.FRAME_SUPER8[0]) * sp.FRAME_SUPER8[1]
    if line.NN(self.l_gate[0]) and line.NN(self.l_gate[2]):
      altw = self.l_gate[2] - self.l_gate[0]
    if line.NN(self.l_gate[1]) and line.NN(self.l_gate[3]):
      alth = self.l_gate[3] - self.l_gate[1]
    if prvw and abs(altw - prvw) < 5:
      best[2] = self.l_gate[2]
      curw = best[2] - best[0]
      asph = (curw / sp.FRAME_SUPER8[0]) * sp.FRAME_SUPER8[1]
    if prvh and alth and (abs(alth - prvh) < 5 or prvh - curh > MARGIN / 2):
      best[3] = self.l_gate[3]
      curh = best[3] - best[1]
      aspw = (curh / sp.FRAME_SUPER8[1]) * sp.FRAME_SUPER8[0]
    print("ASPECT", prvw, prvh, curw, curh, aspw, asph, altw, alth)

    print("BOUNDS")
    print(best, "\n", self.b_gate, "\n", self.l_gate, "\n", guess, "\n", previous)
    if (not sprocket.found or abs(sprocket.bounds.height - spr_size) > 5) \
       and abs(curw - prvw) > 5 and abs(aspw - prvw) > MARGIN \
       and abs(curw - aspw) > MARGIN:
      print("MOVE LEFT")
      best[0] = int(best[2] - width)
    else:
      if line.NN(self.b_gate[2]) and line.NN(self.l_gate[2]) \
         and (abs(self.b_gate[2] - self.l_gate[2]) <= 5 or abs(curw - prvw) <= 5 \
              or line.NN((self.b_gate[1]) and line.NN(previous) \
                  and abs(self.b_gate[1] - previous[1]) < 5)) \
         and best[1] + asph < frame.shape[0]:
        print("MOVE BOTTOM")
        best[3] = int(best[1] + asph)
      elif abs(asph - curh) < abs(aspw - curw):
        print("MOVE TOP")
        best[1] = int(best[3] - asph)
      else:
        print("MOVE RIGHT", curw, aspw, curh, asph)
        best[2] = int(best[0] + aspw)
    print("FINAL\n", best)
    self.best = best
    return

  @property
  def bounds(self):
    if self.found:
      return line.Bounds(self.best[:2], ll=self.best[2:4])
    return None
      
  @property
  def found(self):
    return line.NN(self.best)
