#!/usr/bin/env python3
"""
InkPad - a full-bleed digital notebook in the spirit of Bamboo Paper.

The page fills the entire window edge to edge. No side panels, no toolbars
taking up space - just a few floating controls over the paper:

    top left     home | pages | undo | redo | fullscreen
    top right    pen | colour | eraser | duplicate page
    bottom       < | page x/y | ... | >

Run:    python inkpad.py
Build:  pyinstaller --onefile --windowed --name InkPad inkpad.py
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

from PyQt6.QtCore import (QEvent, QPoint, QPointF, QRectF, QSize,
                          QStandardPaths, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QFont, QFontMetrics, QIcon, QImage,
                         QKeySequence, QPageLayout, QPageSize, QPainter,
                         QPainterPath, QPdfWriter, QPen, QPixmap,
                         QPointingDevice, QShortcut)
from PyQt6.QtWidgets import (QApplication, QColorDialog, QFileDialog, QFrame,
                             QGraphicsDropShadowEffect, QGridLayout, QHBoxLayout,
                             QInputDialog, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QPushButton, QRadioButton, QScrollArea, QSlider,
                             QSystemTrayIcon, QTextEdit, QVBoxLayout, QWidget)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

APP_NAME = "HTLC InkPad"
# Folder that this script lives in, so the logo can be found no matter where
# the app is launched from (double-click, shortcut, terminal, etc.)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ICON_PATH = os.path.join(_APP_DIR, "assets", "htlc_logo.png")
PAGE_W, PAGE_H = 2400, 1350        # 16:9 landscape, fills a normal screen
TEMPLATES = ["blank", "lined", "grid", "dotted", "music"]
TOOLS = ["pen", "pencil", "brush", "highlighter", "text"]

PAPER = "#fdfdfb"
INK_UI = "#3c4149"

PALETTE = [
    "#1c1c1c", "#5a6472", "#9aa4b2", "#ffffff",
    "#24406e", "#1f6fd0", "#0fa3a3", "#2fa84f",
    "#f2c200", "#f08b1d", "#d3455b", "#7b3fb5",
]

COVER_COLORS = ["#1f8fd0", "#2fa84f", "#d3455b", "#7b3fb5",
                "#f08b1d", "#0fa3a3", "#c2185b", "#3b4a5a"]

TOOL_DEFAULTS = {
    "pen":         {"width": 3.0,  "color": "#24406e"},
    "pencil":      {"width": 3.5,  "color": "#5a6472"},
    "brush":       {"width": 9.0,  "color": "#1f6fd0"},
    "highlighter": {"width": 18.0, "color": "#f2c200"},
    "text":        {"width": 24.0, "color": "#1c1c1c"},
    "eraser":      {"width": 26.0, "color": "#ffffff"},
}

_APPDATA_ROOT = (QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
                 or os.path.join(os.path.expanduser("~"), f".{APP_NAME.lower().replace(' ', '_')}"))
LIBRARY_DIR = os.path.join(_APPDATA_ROOT, APP_NAME)
LIBRARY_PATH = os.path.join(LIBRARY_DIR, "library.inkpad")
os.makedirs(LIBRARY_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

@dataclass
class TextElement:
    text: str
    x: float
    y: float
    color: str
    size: float

    def to_dict(self):
        return {"text": self.text, "x": round(self.x, 2), "y": round(self.y, 2),
                "color": self.color, "size": self.size}

    @staticmethod
    def from_dict(d):
        return TextElement(d["text"], d["x"], d["y"], d["color"], d.get("size", 24.0))


@dataclass
class Stroke:
    tool: str
    color: str
    width: float
    points: list = field(default_factory=list)      # [[x, y, pressure], ...]

    def to_dict(self):
        return {"tool": self.tool, "color": self.color, "width": self.width,
                "points": [[round(x, 2), round(y, 2), round(p, 3)]
                           for x, y, p in self.points]}

    @staticmethod
    def from_dict(d):
        return Stroke(d["tool"], d["color"], d["width"],
                      [list(p) for p in d["points"]])


@dataclass
class Page:
    template: str = "blank"
    strokes: list = field(default_factory=list)
    texts: list = field(default_factory=list)

    def to_dict(self):
        return {"template": self.template,
                "strokes": [s.to_dict() for s in self.strokes],
                "texts": [t.to_dict() for t in self.texts]}

    @staticmethod
    def from_dict(d):
        return Page(d.get("template", "blank"),
                    [Stroke.from_dict(s) for s in d.get("strokes", [])],
                    [TextElement.from_dict(t) for t in d.get("texts", [])])


@dataclass
class Notebook:
    name: str = "Notes"
    pages: list = field(default_factory=lambda: [Page()])
    cover: str = COVER_COLORS[0]

    def to_dict(self):
        return {"name": self.name, "cover": self.cover,
                "pages": [p.to_dict() for p in self.pages]}

    @staticmethod
    def from_dict(d):
        nb = Notebook(d.get("name", "Notes"),
                      [Page.from_dict(p) for p in d.get("pages", [])],
                      d.get("cover", COVER_COLORS[0]))
        if not nb.pages:
            nb.pages = [Page()]
        return nb


# ----------------------------------------------------------------------------
# Ink & Text rendering
# ----------------------------------------------------------------------------

def _smooth(points):
    """3-point moving average so hand jitter never reaches the paper."""
    if len(points) < 3:
        return points
    out = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        out.append([(a[0] + b[0] * 2 + c[0]) / 4.0,
                    (a[1] + b[1] * 2 + c[1]) / 4.0,
                    (a[2] + b[2] * 2 + c[2]) / 4.0])
    out.append(points[-1])
    return out


def _catmull_rom_resample(points, step=2.5):
    """Turn a handful of denoised mouse/stylus samples into a dense set of
    points that trace a smooth curve (a Catmull-Rom spline) through them."""
    if len(points) < 3:
        return points
    pts = [points[0]] + points + [points[-1]]
    out = [points[0]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, min(24, int(seg_len / step)))
        for j in range(1, steps + 1):
            t = j / steps
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            pr = p1[2] + (p2[2] - p1[2]) * t
            out.append([x, y, pr])
    return out


def draw_stroke(painter: QPainter, s: Stroke):
    if not s.points or s.tool == "eraser":
        return

    pts = _catmull_rom_resample(_smooth(s.points))
    color = QColor(s.color)

    if s.tool == "highlighter":
        color.setAlphaF(0.32)
        path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
        for i in range(1, len(pts)):
            path.lineTo(QPointF(pts[i][0], pts[i][1]))
        pen = QPen(color, s.width * 2.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        return

    if s.tool == "pen":
        lo, hi, alpha = 0.40, 1.00, 1.00
    elif s.tool == "pencil":
        lo, hi, alpha = 0.55, 1.05, 0.62
    else:
        lo, hi, alpha = 0.10, 1.90, 0.92

    color.setAlphaF(alpha)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if len(pts) == 1:
        x, y, p = pts[0]
        w = s.width * (lo + (hi - lo) * p)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(x, y), w / 2, w / 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        return

    rng = random.Random(int(pts[0][0] * 131 + pts[0][1] * 17))

    for i in range(1, len(pts)):
        x0, y0, p0 = pts[i - 1]
        x1, y1, p1 = pts[i]
        p = max(0.02, (p0 + p1) * 0.5)
        w = max(0.4, s.width * (lo + (hi - lo) * p))

        pen = QPen(color, w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        if s.tool == "pencil":
            grain = QColor(color)
            grain.setAlphaF(alpha * 0.35)
            painter.setPen(QPen(grain, max(0.6, w * 0.35),
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for _ in range(2):
                painter.drawPoint(QPointF(x1 + (rng.random() - .5) * w * 1.5,
                                          y1 + (rng.random() - .5) * w * 1.5))


def draw_text_element(painter: QPainter, t: TextElement):
    if not t.text:
        return
    painter.save()
    font = QFont("Segoe UI" if sys.platform.startswith("win") else "Sans Serif")
    font.setPointSizeF(t.size)
    painter.setFont(font)
    painter.setPen(QColor(t.color))
    painter.drawText(QPointF(t.x, t.y), t.text)
    painter.restore()


def draw_template(painter: QPainter, template: str):
    painter.fillRect(0, 0, PAGE_W, PAGE_H, QColor(PAPER))
    if template == "blank":
        return

    if template == "lined":
        painter.setPen(QPen(QColor("#cfd8e3"), 1.6))
        y = 120
        while y < PAGE_H - 50:
            painter.drawLine(70, y, PAGE_W - 70, y)
            y += 62

    elif template == "grid":
        painter.setPen(QPen(QColor("#dbe2ea"), 1.4))
        for x in range(0, PAGE_W + 1, 50):
            painter.drawLine(x, 0, x, PAGE_H)
        for y in range(0, PAGE_H + 1, 50):
            painter.drawLine(0, y, PAGE_W, y)

    elif template == "dotted":
        painter.setPen(QPen(QColor("#c3ccd8"), 3.6, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        for x in range(52, PAGE_W, 52):
            for y in range(52, PAGE_H, 52):
                painter.drawPoint(x, y)

    elif template == "music":
        painter.setPen(QPen(QColor("#b9c3cf"), 1.5))
        y = 130
        while y < PAGE_H - 110:
            for k in range(5):
                painter.drawLine(90, y + k * 15, PAGE_W - 90, y + k * 15)
            y += 155


def render_page(page: Page, target: QImage = None) -> QImage:
    img = target if target is not None else QImage(
        PAGE_W, PAGE_H, QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw_template(p, page.template)
    for s in page.strokes:
        draw_stroke(p, s)
    for txt in page.texts:
        draw_text_element(p, txt)
    p.end()
    return img


# ----------------------------------------------------------------------------
# Vector icons
# ----------------------------------------------------------------------------

def _p(r, x, y):
    return QPointF(r.x() + x * r.width(), r.y() + y * r.height())


def _rect(r, x, y, w, h):
    return QRectF(r.x() + x * r.width(), r.y() + y * r.height(),
                  w * r.width(), h * r.height())


def paint_icon(p: QPainter, name: str, r: QRectF, color: QColor):
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color, max(1.3, r.width() * 0.085))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    if name == "redo":
        p.translate(r.center())
        p.scale(-1, 1)
        p.translate(-r.center())
        name = "undo"

    if name == "home":
        p.drawPolyline([_p(r, .06, .48), _p(r, .5, .10), _p(r, .94, .48)])
        p.drawPolyline([_p(r, .19, .40), _p(r, .19, .90),
                        _p(r, .81, .90), _p(r, .81, .40)])

    elif name == "pages":
        p.drawRoundedRect(_rect(r, .10, .16, .80, .68), r.width() * .08, r.width() * .08)
        p.setBrush(color)
        p.drawRoundedRect(_rect(r, .19, .58, .26, .18), r.width() * .04, r.width() * .04)

    elif name == "undo":
        box = _rect(r, .12, .26, .76, .62)
        path = QPainterPath()
        path.arcMoveTo(box, 180)
        path.arcTo(box, 180, -180)
        p.drawPath(path)
        p.drawLine(_p(r, .88, .57), _p(r, .88, .84))
        p.drawPolyline([_p(r, .00, .40), _p(r, .12, .57), _p(r, .26, .43)])

    elif name == "fullscreen":
        for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x0 = .10 + sx * .60
            y0 = .10 + sy * .60
            dx = .20 if sx == 0 else -.20
            dy = .20 if sy == 0 else -.20
            p.drawPolyline([_p(r, x0 + (0 if sx == 0 else .20), y0 + dy),
                            _p(r, x0 + (0 if sx == 0 else .20), y0),
                            _p(r, x0 + (dx if sx == 0 else .20 + dx), y0)])

    elif name == "pen":
        path = QPainterPath(_p(r, .14, .88))
        path.lineTo(_p(r, .26, .60))
        path.lineTo(_p(r, .64, .12))
        path.lineTo(_p(r, .86, .32))
        path.lineTo(_p(r, .44, .74))
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(_p(r, .26, .60), _p(r, .44, .74))

    elif name == "pencil":
        path = QPainterPath(_p(r, .12, .88))
        path.lineTo(_p(r, .24, .58))
        path.lineTo(_p(r, .66, .14))
        path.lineTo(_p(r, .88, .34))
        path.lineTo(_p(r, .44, .76))
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(_p(r, .55, .25), _p(r, .77, .45))

    elif name == "brush":
        path = QPainterPath(_p(r, .08, .92))
        path.cubicTo(_p(r, .34, .74), _p(r, .56, .52), _p(r, .92, .10))
        path.cubicTo(_p(r, .62, .58), _p(r, .40, .80), _p(r, .22, .96))
        path.closeSubpath()
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)

    elif name == "highlighter":
        p.drawPolyline([_p(r, .20, .60), _p(r, .58, .14), _p(r, .84, .38),
                        _p(r, .46, .82), _p(r, .20, .82)])
        p.drawLine(_p(r, .12, .94), _p(r, .88, .94))

    elif name == "text":
        f = p.font()
        f.setPixelSize(int(r.height() * 0.82))
        f.setBold(True)
        p.setFont(f)
        p.setPen(color)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Aa")

    elif name == "eraser":
        path = QPainterPath(_p(r, .12, .66))
        path.lineTo(_p(r, .50, .16))
        path.lineTo(_p(r, .88, .48))
        path.lineTo(_p(r, .50, .92))
        path.closeSubpath()
        p.drawPath(path)
        p.drawLine(_p(r, .32, .42), _p(r, .70, .72))

    elif name == "duplicate":
        p.drawRoundedRect(_rect(r, .28, .10, .60, .60), r.width() * .07, r.width() * .07)
        p.drawRoundedRect(_rect(r, .12, .30, .60, .60), r.width() * .07, r.width() * .07)

    elif name == "left":
        p.drawPolyline([_p(r, .62, .16), _p(r, .34, .50), _p(r, .62, .84)])

    elif name == "right":
        p.drawPolyline([_p(r, .38, .16), _p(r, .66, .50), _p(r, .38, .84)])

    elif name == "dots":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        for x in (.22, .50, .78):
            p.drawEllipse(_p(r, x, .50), r.width() * .075, r.width() * .075)

    elif name == "trash":
        p.drawLine(_p(r, .12, .26), _p(r, .88, .26))
        p.drawPolyline([_p(r, .36, .26), _p(r, .36, .13),
                        _p(r, .64, .13), _p(r, .64, .26)])
        p.drawPolyline([_p(r, .22, .26), _p(r, .28, .90),
                        _p(r, .72, .90), _p(r, .78, .26)])

    elif name == "share":
        p.drawPolyline([_p(r, .32, .34), _p(r, .12, .34), _p(r, .12, .90),
                        _p(r, .88, .90), _p(r, .88, .34), _p(r, .68, .34)])
        p.drawLine(_p(r, .50, .62), _p(r, .50, .10))
        p.drawPolyline([_p(r, .32, .28), _p(r, .50, .10), _p(r, .68, .28)])

    elif name == "rename":
        p.drawRoundedRect(_rect(r, .10, .28, .80, .44), r.width() * .10, r.width() * .10)
        p.drawLine(_p(r, .26, .50), _p(r, .62, .50))

    elif name == "copy":
        p.drawRoundedRect(_rect(r, .30, .10, .58, .58), r.width() * .07, r.width() * .07)
        p.drawRoundedRect(_rect(r, .12, .32, .58, .58), r.width() * .07, r.width() * .07)

    elif name == "save":
        p.drawPolyline([_p(r, .14, .62), _p(r, .14, .88), _p(r, .86, .88), _p(r, .86, .62)])
        p.drawLine(_p(r, .50, .10), _p(r, .50, .66))
        p.drawPolyline([_p(r, .30, .46), _p(r, .50, .68), _p(r, .70, .46)])

    elif name == "open":
        p.drawPolyline([_p(r, .10, .80), _p(r, .10, .22), _p(r, .40, .22),
                        _p(r, .50, .36), _p(r, .88, .36)])
        p.drawPolyline([_p(r, .10, .80), _p(r, .24, .48), _p(r, .96, .48), _p(r, .82, .80)])
        p.drawLine(_p(r, .10, .80), _p(r, .82, .80))

    elif name in ("info", "help"):
        p.drawEllipse(_rect(r, .08, .08, .84, .84))
        f = p.font()
        f.setPixelSize(int(r.height() * 0.56))
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "i" if name == "info" else "?")

    elif name == "plus":
        p.drawLine(_p(r, .50, .12), _p(r, .50, .88))
        p.drawLine(_p(r, .12, .50), _p(r, .88, .50))

    elif name == "grid":
        for gx in (0, 1):
            for gy in (0, 1):
                p.drawRoundedRect(_rect(r, .10 + gx * .46, .10 + gy * .46, .34, .34),
                                  r.width() * .05, r.width() * .05)

    elif name == "close":
        p.drawLine(_p(r, .20, .20), _p(r, .80, .80))
        p.drawLine(_p(r, .80, .20), _p(r, .20, .80))

    elif name == "screen":
        p.drawRoundedRect(_rect(r, .08, .14, .84, .58), r.width() * .06, r.width() * .06)
        p.drawLine(_p(r, .50, .72), _p(r, .50, .88))
        p.drawLine(_p(r, .30, .88), _p(r, .70, .88))

    p.restore()


class IconButton(QPushButton):
    """Flat circular button that paints a vector icon."""
    doubleClicked = pyqtSignal()

    def __init__(self, name, tip="", size=40, parent=None):
        super().__init__(parent)
        self.name = name
        self.setToolTip(tip)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setStyleSheet("border:none;background:transparent;")
        self._hover = False

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0, 0, self.width(), self.height())
        if self.isChecked():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#22262d"))
            p.drawEllipse(r.adjusted(2, 2, -2, -2))
            color = QColor("#ffffff")
        else:
            if self._hover:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 18))
                p.drawEllipse(r.adjusted(2, 2, -2, -2))
            color = QColor(INK_UI)
            if not self.isEnabled():
                color.setAlpha(70)
        inset = self.width() * 0.29
        paint_icon(p, self.name, r.adjusted(inset, inset, -inset, -inset), color)
        p.end()


class ColorButton(QPushButton):
    """The filled colour dot in the top-right cluster."""

    def __init__(self, size=40, parent=None):
        super().__init__(parent)
        self.color = "#d3455b"
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setStyleSheet("border:none;background:transparent;")

    def set_color(self, c):
        self.color = c
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0, 0, self.width(), self.height()).adjusted(4, 4, -4, -4)
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        p.setBrush(QColor(self.color))
        p.drawEllipse(r)
        p.end()


class Popover(QFrame):
    """Small floating panel anchored under a button."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("popover")
        self.setStyleSheet("""
            QFrame#popover{background:#ffffff;border:1px solid #e3e6ea;border-radius:14px;}
            QLabel{color:#6b7280;font-size:11px;}
        """)
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(28)
        sh.setOffset(0, 6)
        sh.setColor(QColor(0, 0, 0, 55))
        self.setGraphicsEffect(sh)
        self._last_shown = 0.0
        self.hide()

    def pop_under(self, button, align_right=True):
        pos = button.mapTo(self.parentWidget(), QPoint(0, button.height() + 6))
        x = pos.x() + button.width() - self.width() if align_right else pos.x()
        x = max(10, min(self.parentWidget().width() - self.width() - 10, x))
        self.move(x, pos.y())
        self._last_shown = time.time()
        self.show()
        self.raise_()

    def pop_beside(self, button, gap=10):
        """Anchor this popover to the left of `button`, vertically centred
        on it. Used for the middle-right button column so the panel always
        opens toward open canvas space instead of potentially spilling off
        the bottom of the screen the way 'pop_under' would for a button
        sitting in the vertical middle of the window."""
        parent = self.parentWidget()
        pos = button.mapTo(parent, QPoint(0, 0))
        x = pos.x() - self.width() - gap
        x = max(10, x)
        y = pos.y() + button.height() // 2 - self.height() // 2
        y = max(10, min(parent.height() - self.height() - 10, y))
        self.move(x, y)
        self._last_shown = time.time()
        self.show()
        self.raise_()


# ----------------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------------

class Canvas(QWidget):
    edited = pyqtSignal()
    pageTurnRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.page = Page()
        self.tool = "pen"
        self.color = TOOL_DEFAULTS["pen"]["color"]
        self.pen_width = TOOL_DEFAULTS["pen"]["width"]
        self.eraser_mode = "object"

        self.scale = 1.0
        self.min_scale = 0.1
        self.offset = QPointF(0, 0)

        self.cache = QImage(PAGE_W, PAGE_H, QImage.Format.Format_ARGB32_Premultiplied)
        self.current = None
        self.erased_batch = None
        self._panning = False
        self._pan_anchor = QPointF()
        self._space = False
        self._tablet_active = False
        self._last_pos = None
        self._sim_pressure = 0.6
        self._force_eraser = False
        self._overscroll = 0.0
        self._overscroll_dir = 0

        self.text_editor = None
        self.text_anchor_page = None

        self.undo_stack = []
        self.redo_stack = []
        self.refresh_cache()

    def set_page(self, page: Page):
        self.commit_text()
        self.page = page
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh_cache()
        self.fill_view()

    def refresh_cache(self):
        render_page(self.page, self.cache)
        self.update()

    def _blit(self, stroke):
        p = QPainter(self.cache)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        draw_stroke(p, stroke)
        p.end()

    def thumbnail(self, w=176, h=99):
        return QPixmap.fromImage(self.cache).scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def fill_scale(self):
        if self.width() < 10 or self.height() < 10:
            return 1.0
        return max(self.width() / PAGE_W, self.height() / PAGE_H)

    def fill_view(self):
        self.min_scale = self.fill_scale()
        self.set_scale(self.min_scale, center=True)

    def set_scale(self, s, center=False, anchor=None):
        self.min_scale = self.fill_scale()
        s = max(self.min_scale, min(8.0, s))
        if center or anchor is None:
            self.scale = s
            self.offset = QPointF((self.width() - PAGE_W * s) / 2,
                                  (self.height() - PAGE_H * s) / 2)
        else:
            page_pt = (anchor - self.offset) / self.scale
            self.scale = s
            self.offset = anchor - page_pt * s
        self._clamp()
        self._update_text_editor_position()
        self.update()

    def _clamp(self):
        w, h = PAGE_W * self.scale, PAGE_H * self.scale
        x = min(0.0, max(self.width() - w, self.offset.x())) if w >= self.width() \
            else (self.width() - w) / 2
        y = min(0.0, max(self.height() - h, self.offset.y())) if h >= self.height() \
            else (self.height() - h) / 2
        self.offset = QPointF(x, y)

    def to_page(self, pos: QPointF) -> QPointF:
        return (pos - self.offset) / self.scale

    def to_screen(self, pos: QPointF) -> QPointF:
        return pos * self.scale + self.offset

    def resizeEvent(self, e):
        was_filled = abs(self.scale - self.min_scale) < 1e-6 or e.oldSize().width() <= 0
        self.min_scale = self.fill_scale()
        if was_filled:
            self.set_scale(self.min_scale, center=True)
        else:
            self._clamp()
            self._update_text_editor_position()
        super().resizeEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PAPER))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.save()
        p.translate(self.offset)
        p.scale(self.scale, self.scale)
        p.drawImage(0, 0, self.cache)
        if self.current and self.current.points:
            draw_stroke(p, self.current)
        p.restore()
        p.end()

    def start_text_input(self, page_pos: QPointF):
        self.commit_text()

        self.text_anchor_page = page_pos
        self.text_editor = QLineEdit(self)
        self.text_editor.setFrame(False)

        font_size_screen = max(12, int(self.pen_width * self.scale))
        font = QFont("Segoe UI" if sys.platform.startswith("win") else "Sans Serif")
        font.setPixelSize(font_size_screen)

        self.text_editor.setFont(font)
        self.text_editor.setStyleSheet(
            f"background:transparent; color:{self.color}; border:1px dashed #3b82f6;")

        self._update_text_editor_position()
        self.text_editor.show()
        self.text_editor.setFocus()
        self.text_editor.editingFinished.connect(self.commit_text)

    def _update_text_editor_position(self):
        if self.text_editor and self.text_anchor_page:
            screen_pos = self.to_screen(self.text_anchor_page)
            h = max(24, int(self.pen_width * 1.5 * self.scale))
            self.text_editor.setGeometry(int(screen_pos.x()), int(screen_pos.y() - h * 0.8),
                                         350, h)

    def commit_text(self):
        if not self.text_editor:
            return
        val = self.text_editor.text().strip()
        if val and self.text_anchor_page:
            txt_elem = TextElement(val, self.text_anchor_page.x(), self.text_anchor_page.y(),
                                   self.color, self.pen_width)
            self.page.texts.append(txt_elem)
            self.undo_stack.append(("add_text", txt_elem))
            self.redo_stack.clear()
            self.refresh_cache()
            self.edited.emit()

        self.text_editor.deleteLater()
        self.text_editor = None
        self.text_anchor_page = None

    def _active_tool(self):
        return "eraser" if self._force_eraser else self.tool

    def _begin(self, pos, pressure):
        if self.tool == "text":
            self.start_text_input(pos)
            return

        self.commit_text()

        if self._active_tool() == "eraser":
            self.erased_batch = []
            self._erase_at(pos)
            return
        self.current = Stroke(self._active_tool(), self.color, self.pen_width,
                              [[pos.x(), pos.y(), pressure]])
        self.update()

    def _extend(self, pos, pressure):
        if self.tool == "text":
            return
        if self._active_tool() == "eraser":
            if self.erased_batch is not None:
                self._erase_at(pos)
            return
        if not self.current:
            return
        last = self.current.points[-1]
        if (pos.x() - last[0]) ** 2 + (pos.y() - last[1]) ** 2 < 1.2:
            return
        self.current.points.append([pos.x(), pos.y(), pressure])
        self.update()

    def _end(self):
        if self.tool == "text":
            return
        if self._active_tool() == "eraser":
            if self.erased_batch:
                self.undo_stack.append(("erase_state", list(self.erased_batch)))
                self.redo_stack.clear()
                self.edited.emit()
            self.erased_batch = None
            return
        if self.current and self.current.points:
            self.page.strokes.append(self.current)
            self._blit(self.current)
            self.undo_stack.append(("add", self.current))
            self.redo_stack.clear()
            self.edited.emit()
        self.current = None
        self.update()

    def _erase_at(self, pos):
        r = self.pen_width * 1.1
        r_sq = r * r

        hit_text = [t for t in self.page.texts if (t.x - pos.x()) ** 2 + (t.y - pos.y()) ** 2 <= r_sq * 4]
        if hit_text:
            if not self.erased_batch:
                self.erased_batch = (list(self.page.strokes), list(self.page.texts))
            for t in hit_text:
                self.page.texts.remove(t)
            self.refresh_cache()

        if self.eraser_mode == "object":
            hit = []
            for idx in range(len(self.page.strokes) - 1, -1, -1):
                for (x, y, _p) in self.page.strokes[idx].points:
                    if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= r_sq:
                        hit.append(idx)
                        break
            if hit:
                if not self.erased_batch:
                    self.erased_batch = (list(self.page.strokes), list(self.page.texts))
                for idx in sorted(hit, reverse=True):
                    self.page.strokes.pop(idx)
                self.refresh_cache()

        elif self.eraser_mode == "pixel":
            modified = False
            new_strokes = []
            for s in self.page.strokes:
                segments = []
                cur_seg = []
                for pt in s.points:
                    x, y, _ = pt
                    if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= r_sq:
                        if cur_seg:
                            segments.append(cur_seg)
                            cur_seg = []
                        modified = True
                    else:
                        cur_seg.append(pt)
                if cur_seg:
                    segments.append(cur_seg)

                for seg in segments:
                    new_strokes.append(Stroke(s.tool, s.color, s.width, seg))

            if modified:
                if not self.erased_batch:
                    self.erased_batch = (list(self.page.strokes), list(self.page.texts))
                self.page.strokes = new_strokes
                self.refresh_cache()

    def mousePressEvent(self, e):
        if self._tablet_active:
            return
        if e.button() == Qt.MouseButton.MiddleButton or self._space:
            self._panning = True
            self._pan_anchor = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._last_pos = e.position()
            self._sim_pressure = 0.75
            self._begin(self.to_page(e.position()), self._sim_pressure)

    def mouseMoveEvent(self, e):
        if self._tablet_active:
            return
        if self._panning:
            self.offset += e.position() - self._pan_anchor
            self._pan_anchor = e.position()
            self._clamp()
            self._update_text_editor_position()
            self.update()
            return
        if e.buttons() & Qt.MouseButton.LeftButton:
            if self._last_pos is not None:
                v = math.hypot(e.position().x() - self._last_pos.x(),
                               e.position().y() - self._last_pos.y()) / max(self.scale, .01)
                target = max(0.30, min(1.0, 1.0 - v / 55.0))
                self._sim_pressure += (target - self._sim_pressure) * 0.30
            self._last_pos = e.position()
            self._extend(self.to_page(e.position()), self._sim_pressure)

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._tablet_active:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._end()
            self._last_pos = None

    def tabletEvent(self, e):
        e.accept()
        pos = self.to_page(e.position())
        pressure = max(0.02, e.pressure()) if e.pressure() > 0 else 0.6
        try:
            self._force_eraser = (e.pointingDevice().pointerType()
                                  == QPointingDevice.PointerType.Eraser)
        except Exception:
            self._force_eraser = False

        t = e.type()
        if t == QEvent.Type.TabletPress:
            self._tablet_active = True
            self._begin(pos, pressure)
        elif t == QEvent.Type.TabletMove and self._tablet_active:
            self._extend(pos, pressure)
        elif t == QEvent.Type.TabletRelease:
            self._end()
            self._tablet_active = False
            self._force_eraser = False

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_scale(self.scale * (1.0015 ** e.angleDelta().y()), anchor=e.position())
            e.accept()
            return

        pd = e.pixelDelta()
        if not pd.isNull():
            dx, dy = float(pd.x()), float(pd.y())
        else:
            dx, dy = e.angleDelta().x() * .6, e.angleDelta().y() * .6

        if dx == 0.0 and dy != 0.0 and (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            dx, dy = dy, 0.0

        page_h = PAGE_H * self.scale
        at_top = self.offset.y() >= -0.5
        at_bottom = self.offset.y() <= (self.height() - page_h) + 0.5

        self.offset += QPointF(dx, dy)
        self._clamp()
        self._update_text_editor_position()
        self.update()

        pushing_past_top = at_top and dy > 0
        pushing_past_bottom = at_bottom and dy < 0 and page_h >= self.height()

        if pushing_past_top or pushing_past_bottom:
            self._overscroll += abs(dy)
            self._overscroll_dir = -1 if pushing_past_top else 1
        else:
            self._overscroll = 0.0

        if self._overscroll > 160:
            self._overscroll = 0.0
            self.pageTurnRequested.emit(self._overscroll_dir)

        e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().keyReleaseEvent(e)

    def undo(self):
        self.commit_text()
        if not self.undo_stack:
            return
        action, data = self.undo_stack.pop()
        if action == "add" and data in self.page.strokes:
            self.page.strokes.remove(data)
            self.redo_stack.append((action, data))
        elif action == "add_text" and data in self.page.texts:
            self.page.texts.remove(data)
            self.redo_stack.append((action, data))
        elif action == "erase_state":
            current_state = (list(self.page.strokes), list(self.page.texts))
            self.page.strokes, self.page.texts = list(data[0]), list(data[1])
            self.redo_stack.append(("erase_state", current_state))
        elif action == "clear":
            self.page.strokes, self.page.texts = list(data[0]), list(data[1])
            self.redo_stack.append((action, data))
        self.refresh_cache()
        self.edited.emit()

    def redo(self):
        self.commit_text()
        if not self.redo_stack:
            return
        action, data = self.redo_stack.pop()
        if action == "add":
            self.page.strokes.append(data)
            self.undo_stack.append((action, data))
        elif action == "add_text":
            self.page.texts.append(data)
            self.undo_stack.append((action, data))
        elif action == "erase_state":
            current_state = (list(self.page.strokes), list(self.page.texts))
            self.page.strokes, self.page.texts = list(data[0]), list(data[1])
            self.undo_stack.append(("erase_state", current_state))
        elif action == "clear":
            self.page.strokes, self.page.texts = [], []
            self.undo_stack.append((action, data))
        self.refresh_cache()
        self.edited.emit()

    def clear_page(self):
        self.commit_text()
        if not self.page.strokes and not self.page.texts:
            return
        self.undo_stack.append(("clear", (list(self.page.strokes), list(self.page.texts))))
        self.redo_stack.clear()
        self.page.strokes = []
        self.page.texts = []
        self.refresh_cache()
        self.edited.emit()


# ----------------------------------------------------------------------------
# Screen overlay
# ----------------------------------------------------------------------------

class OverlayCanvas(QWidget):
    def __init__(self, on_escape=None):
        super().__init__()
        self.on_escape = on_escape
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(QApplication.primaryScreen().geometry())

        self.background = None
        self.strokes = []
        self.current = None
        self.tool = "pen"
        self.color = "#d3455b"
        self.pen_width = 4.0
        self._force_eraser = False
        self._tablet_active = False

    def freeze(self):
        self.background = QApplication.primaryScreen().grabWindow(0)
        self.strokes = []
        self.current = None

    def discard(self):
        self.background = None
        self.strokes = []
        self.current = None

    def flattened(self):
        pm = QPixmap(self.background) if self.background is not None else self.grab()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for s in self.strokes:
            draw_stroke(p, s)
        p.end()
        return pm

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.background is not None:
            p.drawPixmap(0, 0, self.background)
        for s in self.strokes:
            draw_stroke(p, s)
        if self.current and self.current.points:
            draw_stroke(p, self.current)
        p.end()

    def clear(self):
        self.strokes = []
        self.current = None
        self.update()

    def _active_tool(self):
        return "eraser" if self._force_eraser else self.tool

    def _begin(self, pos, pressure):
        if self._active_tool() == "eraser":
            self._erase_at(pos)
            return
        self.current = Stroke(self._active_tool(), self.color, self.pen_width,
                              [[pos.x(), pos.y(), pressure]])
        self.update()

    def _extend(self, pos, pressure):
        if self._active_tool() == "eraser":
            self._erase_at(pos)
            return
        if not self.current:
            return
        self.current.points.append([pos.x(), pos.y(), pressure])
        self.update()

    def _end(self):
        if self.current and self.current.points:
            self.strokes.append(self.current)
        self.current = None
        self.update()

    def _erase_at(self, pos):
        r = self.pen_width * 1.4
        self.strokes = [s for s in self.strokes if all(
            (x - pos.x()) ** 2 + (y - pos.y()) ** 2 > r * r for x, y, _p in s.points)]
        self.update()

    def mousePressEvent(self, e):
        if self._tablet_active or e.button() != Qt.MouseButton.LeftButton:
            return
        self._begin(e.position(), 0.75)

    def mouseMoveEvent(self, e):
        if self._tablet_active or not (e.buttons() & Qt.MouseButton.LeftButton):
            return
        self._extend(e.position(), 0.75)

    def mouseReleaseEvent(self, e):
        if self._tablet_active or e.button() != Qt.MouseButton.LeftButton:
            return
        self._end()

    def tabletEvent(self, e):
        e.accept()
        pressure = max(0.02, e.pressure()) if e.pressure() > 0 else 0.6
        try:
            self._force_eraser = (e.pointingDevice().pointerType()
                                  == QPointingDevice.PointerType.Eraser)
        except Exception:
            self._force_eraser = False

        t = e.type()
        if t == QEvent.Type.TabletPress:
            self._tablet_active = True
            self._begin(e.position(), pressure)
        elif t == QEvent.Type.TabletMove and self._tablet_active:
            self._extend(e.position(), pressure)
        elif t == QEvent.Type.TabletRelease:
            self._end()
            self._tablet_active = False
            self._force_eraser = False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.on_escape:
            self.on_escape()
        super().keyPressEvent(e)


class OverlayToolbar(QFrame):
    def __init__(self, overlay, on_close):
        super().__init__(overlay)
        self.overlay = overlay
        self.on_close = on_close
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setObjectName("overlayBar")
        self.setStyleSheet("QFrame#overlayBar{background:#22262d;border-radius:16px;}")
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(24)
        sh.setOffset(0, 6)
        sh.setColor(QColor(0, 0, 0, 130))
        self.setGraphicsEffect(sh)
        self._drag_pos = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        def add(name, tip, cb, checkable=False):
            b = IconButton(name, tip, 34, self)
            b.setCheckable(checkable)
            b.clicked.connect(lambda: cb())
            lay.addWidget(b)
            return b

        self.b_pen = add("pen", "Pen", self.pick_pen, checkable=True)
        self.b_pen.setChecked(True)

        self.b_color = ColorButton(28, self)
        self.b_color.set_color(overlay.color)
        self.b_color.clicked.connect(self.pick_color)
        lay.addWidget(self.b_color)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(1, 24)
        self.slider.setValue(int(overlay.pen_width))
        self.slider.setFixedWidth(90)
        self.slider.valueChanged.connect(self.on_size)
        lay.addWidget(self.slider)

        self.b_eraser = add("eraser", "Eraser", self.pick_eraser, checkable=True)
        add("trash", "Clear all ink", overlay.clear)
        add("save", "Save annotated screenshot", self.save_screenshot)
        add("close", "Discard drawing & close (Esc)", self.close_overlay)

        self.adjustSize()

    def pick_pen(self):
        self.overlay.tool = "pen"
        self.b_pen.setChecked(True)
        self.b_eraser.setChecked(False)

    def pick_eraser(self):
        self.overlay.tool = "eraser"
        self.b_eraser.setChecked(True)
        self.b_pen.setChecked(False)

    def pick_color(self):
        c = QColorDialog.getColor(QColor(self.overlay.color), self, "Ink colour")
        if c.isValid():
            self.overlay.color = c.name()
            self.b_color.set_color(c.name())

    def on_size(self, v):
        self.overlay.pen_width = float(v)

    def close_overlay(self):
        self.on_close()

    def save_screenshot(self):
        pm = self.overlay.flattened()
        path, _ = QFileDialog.getSaveFileName(self, "Save annotated screenshot",
                                              "annotation.png", "PNG image (*.png)")
        if path:
            pm.save(path, "PNG")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


# ----------------------------------------------------------------------------
# Notebook view
# ----------------------------------------------------------------------------

class NotebookView(QWidget):
    goHome = pyqtSignal()

    def __init__(self, app_window):
        super().__init__()
        self.app = app_window
        self.canvas = Canvas(self)
        self.canvas.edited.connect(self.on_edited)
        self.canvas.pageTurnRequested.connect(self.app.turn_page_with_scroll)
        self.canvas.installEventFilter(self)

        self._ink_tool = "pen"
        self._ink_color = TOOL_DEFAULTS["pen"]["color"]
        self._ink_width = TOOL_DEFAULTS["pen"]["width"]

        mk = lambda n, tip, cb, size=40: self._button(n, tip, cb, size)

        # top-left cluster
        self.b_home = mk("home", "Library", self.goHome.emit)
        self.b_pages = mk("pages", "Pages", self.toggle_page_grid)
        self.b_undo = mk("undo", "Undo", self.canvas.undo)
        self.b_redo = mk("redo", "Redo", self.canvas.redo)
        self.b_full = mk("fullscreen", "Fullscreen", self.app_fullscreen)
        self.b_overlay = mk("screen", "Annotate Screen", self.app.start_overlay)
        self.left_group = [self.b_home, self.b_pages, self.b_undo, self.b_redo,
                          self.b_full, self.b_overlay]

        # top-right cluster
        self.b_pen = mk("pen", "Pen", self.on_pen)
        self.b_pen.setCheckable(True)
        self.b_pen.setChecked(True)

        self.b_color = ColorButton(40, self)
        self.b_color.setToolTip("Color")
        self.b_color.clicked.connect(self.on_color)

        self.b_eraser = mk("eraser", "Eraser", self.on_eraser)
        self.b_eraser.setCheckable(True)
        self.b_eraser.doubleClicked.connect(self.on_eraser_double_click)

        self.b_dup = mk("duplicate", "Add Page", self.app_add_page)
        self.right_group = [self.b_pen, self.b_color, self.b_eraser, self.b_dup]

        # bottom bar
        self.b_prev = mk("left", "Previous page", lambda: self.app.turn_page(-1), 36)
        self.b_next = mk("right", "Next page", lambda: self.app.turn_page(1), 36)
        self.b_dots = mk("dots", "Page options", self.on_dots, 30)
        self.lbl_page = QLabel("1/1", self)
        self.lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_page.setStyleSheet("color:#9aa2ad;font-size:13px;background:transparent;")
        self.lbl_page.setFixedSize(90, 18)

        self._build_pen_popover()
        self._build_eraser_popover()
        self._build_color_popover()
        self._build_page_menu()
        self._build_page_grid()

        self.b_color.set_color(self.canvas.color)

    def _button(self, name, tip, cb, size=40):
        b = IconButton(name, tip, size, self)
        b.clicked.connect(lambda: cb())
        return b

    def _build_pen_popover(self):
        self.pop_pen = Popover(self)
        lay = QVBoxLayout(self.pop_pen)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.tool_buttons = {}
        for t in TOOLS:
            b = IconButton(t, t.capitalize(), 40, self.pop_pen)
            b.setCheckable(True)
            b.clicked.connect(lambda _, tt=t: self.select_tool(tt))
            row.addWidget(b)
            self.tool_buttons[t] = b
        holder = QWidget(self.pop_pen)
        holder.setLayout(row)
        lay.addWidget(holder)
        lay.addWidget(QLabel("SIZE / FONT SIZE", self.pop_pen))
        self.size_slider = QSlider(Qt.Orientation.Horizontal, self.pop_pen)
        self.size_slider.setRange(1, 72)
        self.size_slider.setValue(int(self.canvas.pen_width))
        self.size_slider.setFixedWidth(184)
        self.size_slider.valueChanged.connect(self.on_size)
        lay.addWidget(self.size_slider)
        self.pop_pen.adjustSize()
        self.tool_buttons["pen"].setChecked(True)

    def _build_eraser_popover(self):
        self.pop_eraser = Popover(self)
        lay = QVBoxLayout(self.pop_eraser)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)

        lay.addWidget(QLabel("ERASER MODE", self.pop_eraser))
        self.rad_obj = QRadioButton("Erase by Object", self.pop_eraser)
        self.rad_pix = QRadioButton("Erase by Pixel", self.pop_eraser)
        self.rad_obj.setChecked(True)
        self.rad_obj.toggled.connect(self.on_eraser_mode_change)

        lay.addWidget(self.rad_obj)
        lay.addWidget(self.rad_pix)

        self.lbl_eraser_size = QLabel("SIZE", self.pop_eraser)
        lay.addWidget(self.lbl_eraser_size)

        self.eraser_size_slider = QSlider(Qt.Orientation.Horizontal, self.pop_eraser)
        self.eraser_size_slider.setRange(5, 100)
        self.eraser_size_slider.setValue(int(TOOL_DEFAULTS["eraser"]["width"]))
        self.eraser_size_slider.setFixedWidth(184)
        self.eraser_size_slider.valueChanged.connect(self.on_eraser_size)
        lay.addWidget(self.eraser_size_slider)

        self.lbl_eraser_size.hide()
        self.eraser_size_slider.hide()

        self.pop_eraser.adjustSize()

    def _build_color_popover(self):
        self.pop_color = Popover(self)
        lay = QVBoxLayout(self.pop_color)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        grid = QGridLayout()
        grid.setSpacing(7)
        self.swatches = []
        for i, c in enumerate(PALETTE):
            b = QPushButton(self.pop_color)
            b.setFixedSize(26, 26)
            b.hex = c
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, cc=c: self.select_color(cc))
            grid.addWidget(b, i // 4, i % 4)
            self.swatches.append(b)
        holder = QWidget(self.pop_color)
        holder.setLayout(grid)
        lay.addWidget(holder)
        more = QPushButton("More colours...", self.pop_color)
        more.setCursor(Qt.CursorShape.PointingHandCursor)
        more.setStyleSheet("QPushButton{border:none;color:#6b7280;font-size:12px;"
                           "padding:4px;}QPushButton:hover{color:#111827;}")
        more.clicked.connect(self.pick_custom_color)
        lay.addWidget(more)
        self.pop_color.adjustSize()
        self._style_swatches()

    def _build_page_menu(self):
        self.pop_page = Popover(self)
        lay = QVBoxLayout(self.pop_page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)

        def item(text, cb):
            b = QPushButton(text, self.pop_page)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("QPushButton{border:none;text-align:left;padding:7px 12px;"
                            "color:#374151;font-size:13px;border-radius:7px;}"
                            "QPushButton:hover{background:#f1f3f6;}")
            b.clicked.connect(lambda: (self.pop_page.hide(), cb()))
            lay.addWidget(b)
            return b

        item("New page", self.app_add_page)
        item("Duplicate page", lambda: self.app.duplicate_page())
        item("Delete page", lambda: self.app.delete_page())
        sep = QFrame(self.pop_page)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#e8eaee;")
        lay.addWidget(sep)
        for t in TEMPLATES:
            item(f"Paper: {t.capitalize()}", lambda tt=t: self.app.set_template(tt))
        sep2 = QFrame(self.pop_page)
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#e8eaee;")
        lay.addWidget(sep2)
        item("Clear page", self.canvas.clear_page)
        item("Export page as PNG...", lambda: self.app.export_png())
        item("Export notebook as PDF...", lambda: self.app.export_pdf())
        self.pop_page.adjustSize()

    def _build_page_grid(self):
        self.page_grid = QWidget(self)
        self.page_grid.setStyleSheet("background:rgba(250,250,248,0.97);")
        lay = QVBoxLayout(self.page_grid)
        lay.setContentsMargins(20, 14, 20, 20)
        top = QHBoxLayout()
        title = QLabel("Pages", self.page_grid)
        title.setStyleSheet("color:#374151;font-size:16px;background:transparent;")
        top.addWidget(title)
        top.addStretch()
        b_add = IconButton("plus", "New page", 36, self.page_grid)
        b_add.clicked.connect(self.app_add_page)
        b_close = IconButton("close", "Close", 36, self.page_grid)
        b_close.clicked.connect(self.toggle_page_grid)
        top.addWidget(b_add)
        top.addWidget(b_close)
        holder = QWidget(self.page_grid)
        holder.setLayout(top)
        lay.addWidget(holder)

        self.grid_list = QListWidget(self.page_grid)
        self.grid_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid_list.setIconSize(QSize(230, 130))
        self.grid_list.setGridSize(QSize(252, 172))
        self.grid_list.setSpacing(10)
        self.grid_list.setMovement(QListWidget.Movement.Static)
        self.grid_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid_list.setStyleSheet(
            "QListWidget{background:transparent;border:none;color:#6b7280;}"
            "QListWidget::item{border-radius:6px;}"
            "QListWidget::item:selected{background:#e6eefb;color:#1f2937;}")
        self.grid_list.itemClicked.connect(self.on_grid_click)
        lay.addWidget(self.grid_list)
        self.page_grid.hide()

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        self.canvas.setGeometry(0, 0, w, h)

        x = 12
        for b in self.left_group:
            b.move(x, 12)
            x += b.width() + 4

        # Right cluster (pen, colour, eraser, add page): stacked vertically
        # and centred on the right edge instead of pinned to the top.
        # On a big touch-screen TV a shorter tutor can't comfortably reach
        # buttons parked at the very top, but the middle of the screen edge
        # is within easy arm's reach for most people standing at the board.
        gap = 10
        total_h = sum(b.height() for b in self.right_group) + gap * (len(self.right_group) - 1)
        y = max(12, h // 2 - total_h // 2)
        right_x = w - 12 - max(b.width() for b in self.right_group)
        for b in self.right_group:
            b.move(right_x, y)
            y += b.height() + gap

        self.b_prev.move(12, h - 48)
        self.b_next.move(w - 48, h - 48)
        self.lbl_page.move(w // 2 - 45, h - 46)
        self.b_dots.move(w // 2 - 15, h - 30)
        self.page_grid.setGeometry(0, 0, w, h)
        for pop in (self.pop_pen, self.pop_eraser, self.pop_color, self.pop_page):
            pop.hide()
        super().resizeEvent(e)

    def eventFilter(self, obj, e):
        if obj is self.canvas and e.type() in (QEvent.Type.MouseButtonPress,
                                               QEvent.Type.TabletPress):
            if self.close_popovers():
                return True
        return super().eventFilter(obj, e)

    def close_popovers(self):
        closed = False
        now = time.time()
        for pop in (self.pop_pen, self.pop_eraser, self.pop_color, self.pop_page):
            if pop.isVisible():
                if now - getattr(pop, "_last_shown", 0.0) < 0.2:
                    continue
                pop.hide()
                closed = True
        return closed

    def on_pen(self):
        self.b_pen.setChecked(True)
        if self.canvas.tool == "eraser":
            self.restore_ink_tool()
            return
        if self.pop_pen.isVisible():
            self.pop_pen.hide()
        else:
            self.close_popovers()
            self.pop_pen.pop_beside(self.b_pen)

    def restore_ink_tool(self):
        self.canvas.tool = self._ink_tool
        self.canvas.color = self._ink_color
        self.canvas.pen_width = self._ink_width
        self.b_color.set_color(self._ink_color)

        self.size_slider.blockSignals(True)
        self.size_slider.setValue(int(self._ink_width))
        self.size_slider.blockSignals(False)

        for t, b in self.tool_buttons.items():
            b.setChecked(t == self._ink_tool)

        self.b_pen.name = self._ink_tool
        self.b_pen.setChecked(True)
        self.b_eraser.setChecked(False)
        self.b_pen.update()
        self._style_swatches()

    def select_tool(self, tool):
        self._ink_tool = tool
        self.canvas.tool = tool
        d = TOOL_DEFAULTS[tool]

        self._ink_width = d["width"]
        self._ink_color = d["color"]

        self.canvas.pen_width = self._ink_width
        self.size_slider.blockSignals(True)
        self.size_slider.setValue(int(self._ink_width))
        self.size_slider.blockSignals(False)

        self.select_color(self._ink_color)

        for t, b in self.tool_buttons.items():
            b.setChecked(t == tool)
        self.b_pen.name = tool
        self.b_pen.setChecked(True)
        self.b_eraser.setChecked(False)
        self.b_pen.update()

    def on_eraser(self):
        self.canvas.commit_text()
        self.canvas.tool = "eraser"
        self.canvas.pen_width = self.eraser_size_slider.value()
        self.b_eraser.setChecked(True)
        self.b_pen.setChecked(False)

    def on_eraser_double_click(self):
        self.on_eraser()
        if self.pop_eraser.isVisible():
            self.pop_eraser.hide()
        else:
            self.close_popovers()
            self.pop_eraser.pop_beside(self.b_eraser)

    def on_eraser_mode_change(self):
        is_object_mode = self.rad_obj.isChecked()
        self.canvas.eraser_mode = "object" if is_object_mode else "pixel"

        self.lbl_eraser_size.setVisible(not is_object_mode)
        self.eraser_size_slider.setVisible(not is_object_mode)
        self.pop_eraser.adjustSize()

    def on_eraser_size(self, v):
        if self.canvas.tool == "eraser":
            self.canvas.pen_width = float(v)

    def on_color(self):
        if self.pop_color.isVisible():
            self.pop_color.hide()
        else:
            self.close_popovers()
            self.pop_color.pop_beside(self.b_color)

    def select_color(self, c):
        self._ink_color = c
        self.canvas.color = c
        self.b_color.set_color(c)
        self._style_swatches()
        if self.canvas.tool == "eraser":
            self.restore_ink_tool()

    def _style_swatches(self):
        for b in self.swatches:
            sel = b.hex.lower() == self.canvas.color.lower()
            self.setUpdatesEnabled(True)
            b.setStyleSheet(
                f"background:{b.hex};border-radius:13px;"
                f"border:{'3px solid #22262d' if sel else '1px solid rgba(0,0,0,0.18)'};")

    def pick_custom_color(self):
        self.pop_color.hide()
        c = QColorDialog.getColor(QColor(self.canvas.color), self, "Ink colour")
        if c.isValid():
            self.select_color(c.name())

    def on_size(self, v):
        self._ink_width = float(v)
        if self.canvas.tool != "eraser":
            self.canvas.pen_width = self._ink_width

    def on_dots(self):
        if self.pop_page.isVisible():
            self.pop_page.hide()
        else:
            self.close_popovers()
            pos = self.b_dots.mapTo(self, QPoint(0, 0))
            self.pop_page.adjustSize()
            x = max(10, min(self.width() - self.pop_page.width() - 10,
                            pos.x() + 15 - self.pop_page.width() // 2))
            self.pop_page.move(x, max(10, pos.y() - self.pop_page.height() - 8))
            self.pop_page.show()
            self.pop_page.raise_()

    def toggle_page_grid(self):
        if self.page_grid.isVisible():
            self.page_grid.hide()
            return
        self.close_popovers()
        self.refresh_grid()
        self.page_grid.show()
        self.page_grid.raise_()

    def refresh_grid(self):
        self.grid_list.clear()
        for i, pg in enumerate(self.app.notebook.pages):
            if i == self.app.page_index:
                pm = self.canvas.thumbnail(230, 130)
            else:
                pm = QPixmap.fromImage(render_page(pg)).scaled(
                    230, 130, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            icon = QIcon()
            icon.addPixmap(pm, QIcon.Mode.Normal)
            icon.addPixmap(pm, QIcon.Mode.Selected)
            icon.addPixmap(pm, QIcon.Mode.Active)
            item = QListWidgetItem(icon, f"{i + 1}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_list.addItem(item)
        self.grid_list.setCurrentRow(self.app.page_index)

    def on_grid_click(self, item):
        self.app.goto_page(self.grid_list.row(item))
        self.page_grid.hide()

    def app_add_page(self):
        self.app.add_page()

    def app_fullscreen(self):
        self.app.toggle_fullscreen()

    def on_edited(self):
        self.app.dirty = True

    def sync(self):
        n = len(self.app.notebook.pages)
        i = self.app.page_index
        self.lbl_page.setText(f"{i + 1}/{n}")
        self.b_prev.setEnabled(i > 0)
        self.b_next.setEnabled(i < n - 1)


# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------

class Cover(QWidget):
    opened = pyqtSignal(int)
    selected = pyqtSignal(int)

    def __init__(self, notebook, index, parent=None):
        super().__init__(parent)
        self.nb = notebook
        self.index = index
        self.is_selected = False
        self.setFixedSize(232, 310)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, e):
        self.selected.emit(self.index)

    def mouseDoubleClickEvent(self, e):
        self.opened.emit(self.index)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(14, 8, self.width() - 28, self.height() - 30)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 28))
        p.drawRoundedRect(r.adjusted(3, 5, 3, 7), 4, 4)

        p.setBrush(QColor(self.nb.cover))
        p.drawRoundedRect(r, 3, 3)
        p.setBrush(QColor(255, 255, 255, 26))
        p.drawRoundedRect(QRectF(r.x(), r.y(), 12, r.height()), 3, 3)

        p.setPen(QColor("#ffffff"))
        f = QFont(p.font())
        f.setPixelSize(21)
        f.setWeight(QFont.Weight.Light)
        p.setFont(f)
        avail = int(r.width()) - 40
        title = QFontMetrics(f).elidedText(self.nb.name, Qt.TextElideMode.ElideRight, avail)
        p.drawText(QRectF(r.x() + 24, r.y() + r.height() * .42, avail, 32),
                   Qt.AlignmentFlag.AlignLeft, title)
        f.setPixelSize(12)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 190))
        n = len(self.nb.pages)
        p.drawText(QRectF(r.x() + 24, r.y() + r.height() * .42 + 32, avail, 20),
                   Qt.AlignmentFlag.AlignLeft, f"{n} page{'s' if n != 1 else ''}")

        if self.is_selected:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#3b82f6"), 2.5))
            p.drawRoundedRect(r.adjusted(-5, -5, 5, 5), 7, 7)
        p.end()


class LibraryView(QWidget):
    opened = pyqtSignal(int)

    def __init__(self, app_window):
        super().__init__()
        self.app = app_window
        self.sel = 0
        self.setStyleSheet("background:#ffffff;")

        bar = QWidget(self)
        bar.setFixedHeight(56)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(4)

        def add(icon, tip, cb, right=False):
            b = IconButton(icon, tip, 38, bar)
            b.clicked.connect(lambda: cb())
            bl.addWidget(b)
            return b

        add("trash", "Delete notebook", self.delete_nb)
        add("share", "Export notebook as PDF", lambda: self.app.export_pdf(self.sel))
        add("rename", "Rename notebook", self.rename_nb)
        add("copy", "Duplicate notebook", self.duplicate_nb)
        bl.addStretch()
        add("save", "Save all notebooks (Ctrl+S)", self.app.save_file)
        add("open", "Open a .inkpad file (Ctrl+O)", self.app.open_file)
        add("screen", "Freeze screen & annotate (Ctrl+Shift+A)", self.app.start_overlay)
        add("help", "Shortcuts & tips", self.app.show_help)
        add("plus", "New notebook", self.app.new_notebook)

        self.area = QScrollArea(self)
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.Shape.NoFrame)
        self.area.setStyleSheet("background:#ffffff;")
        self.inner = QWidget()
        self.inner.setStyleSheet("background:#ffffff;")
        self.grid = QGridLayout(self.inner)
        self.grid.setContentsMargins(40, 30, 40, 40)
        self.grid.setSpacing(18)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.area.setWidget(self.inner)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        hint = QLabel("Double-click a notebook to open it")
        hint.setStyleSheet("color:#9aa2ad;font-size:12px;background:transparent;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.area, 1)
        lay.addWidget(hint)
        lay.addSpacing(10)

    def refresh(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        cols = max(1, min(5, (self.width() - 80) // 250))
        for i, nb in enumerate(self.app.notebooks):
            c = Cover(nb, i, self.inner)
            c.is_selected = (i == self.sel)
            c.opened.connect(self.opened.emit)
            c.selected.connect(self.on_select)
            self.grid.addWidget(c, i // cols, i % cols)

    def on_select(self, i):
        self.sel = i
        self.refresh()

    def resizeEvent(self, e):
        self.refresh()
        super().resizeEvent(e)

    def rename_nb(self):
        nb = self.app.notebooks[self.sel]
        name, ok = QInputDialog.getText(self, "Rename notebook", "Name:", text=nb.name)
        if ok and name:
            nb.name = name
            self.app.dirty = True
            self.refresh()

    def duplicate_nb(self):
        src = self.app.notebooks[self.sel]
        self.app.notebooks.append(Notebook.from_dict(json.loads(json.dumps(src.to_dict()))))
        self.app.notebooks[-1].name = src.name + " copy"
        self.app.dirty = True
        self.refresh()

    def delete_nb(self):
        if len(self.app.notebooks) == 1:
            QMessageBox.information(self, APP_NAME, "You need at least one notebook.")
            return
        nb = self.app.notebooks[self.sel]
        if QMessageBox.question(self, APP_NAME,
                                f"Delete '{nb.name}' and its "
                                f"{len(nb.pages)} page(s)?") != QMessageBox.StandardButton.Yes:
            return
        self.app.notebooks.pop(self.sel)
        self.sel = max(0, self.sel - 1)
        self.app.dirty = True
        self.refresh()


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------

class InkPad(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 860)
        self.setStyleSheet("QMainWindow{background:#ffffff;}")

        self.notebooks = self._load_library()
        self.nb_index = 0
        self.page_index = 0
        self.file_path = None
        self.dirty = False

        self.library = LibraryView(self)
        self.library.opened.connect(self.open_notebook)
        self.view = NotebookView(self)
        self.view.goHome.connect(self.go_home)
        self.canvas = self.view.canvas

        self.stack = QWidget()
        sl = QVBoxLayout(self.stack)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.library)
        sl.addWidget(self.view)
        self.view.hide()
        self.setCentralWidget(self.stack)

        self._overlay = None
        self._overlay_bar = None
        self._init_tray()

        self._shortcuts()
        self.library.refresh()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(30_000)

    @property
    def notebook(self):
        return self.notebooks[self.nb_index]

    @property
    def page(self):
        return self.notebook.pages[self.page_index]

    def _shortcuts(self):
        S = lambda k, fn: QShortcut(QKeySequence(k), self, activated=fn)
        S("Ctrl+Z", lambda: self.canvas.undo())
        S("Ctrl+Y", lambda: self.canvas.redo())
        S("Ctrl+Shift+Z", lambda: self.canvas.redo())
        S("Ctrl+S", self.save_file)
        S("Ctrl+Shift+S", self.save_file_as)
        S("Ctrl+O", self.open_file)
        S("Ctrl+N", self.add_page)
        S("Ctrl+E", self.export_png)
        S("Ctrl+P", lambda: self.export_pdf())
        S("Ctrl+Right", lambda: self.turn_page(1))
        S("Ctrl+Left", lambda: self.turn_page(-1))
        S("Right", lambda: self.turn_page(1))
        S("Left", lambda: self.turn_page(-1))
        S("Ctrl++", lambda: self.canvas.set_scale(self.canvas.scale * 1.25))
        S("Ctrl+-", lambda: self.canvas.set_scale(self.canvas.scale / 1.25))
        S("Ctrl+0", self.canvas.fill_view)
        S("F11", self.toggle_fullscreen)
        S("Ctrl+Shift+A", self.start_overlay)
        S("Esc", self.on_escape)

    def on_escape(self):
        if self.canvas.text_editor:
            self.canvas.commit_text()
        elif self.isFullScreen():
            self.showNormal()
        elif self.view.page_grid.isVisible():
            self.view.toggle_page_grid()
        elif self.view.close_popovers():
            pass
        elif self.view.isVisible():
            self.go_home()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _init_tray(self):
        if os.path.exists(APP_ICON_PATH):
            tray_icon = QIcon(APP_ICON_PATH)
        else:
            pm = QPixmap(64, 64)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            paint_icon(p, "screen", QRectF(4, 4, 56, 56), QColor("#135bec"))
            p.end()
            tray_icon = QIcon(pm)

        self.tray = QSystemTrayIcon(tray_icon, self)
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        menu.addAction("Freeze screen & annotate", self.start_overlay)
        menu.addAction("Close annotation (discard)", self.stop_overlay)
        menu.addSeparator()
        menu.addAction(f"Show {APP_NAME}", self._restore_from_tray)
        menu.addAction("Quit", self.close)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.start_overlay()

    def _restore_from_tray(self):
        self.showNormal()
        self.activateWindow()

    def start_overlay(self):
        self.showMinimized()
        QTimer.singleShot(150, self._show_overlay_after_minimize)

    def _show_overlay_after_minimize(self):
        if self._overlay is None:
            self._overlay = OverlayCanvas(on_escape=self.stop_overlay)
            self._overlay_bar = OverlayToolbar(self._overlay, self.stop_overlay)

        self._overlay.freeze()
        self._overlay.show()
        self._overlay.setFocus()
        self._overlay.activateWindow()

        bar = self._overlay_bar
        bar.adjustSize()
        screen = QApplication.primaryScreen().geometry()
        bar.move(screen.center().x() - bar.width() // 2, screen.top() + 24)
        bar.show()
        bar.raise_()

    def stop_overlay(self):
        if self._overlay:
            self._overlay.hide()
            self._overlay.discard()
        if self._overlay_bar:
            self._overlay_bar.hide()
        self.showNormal()
        self.activateWindow()

    def open_notebook(self, idx):
        self.nb_index = idx
        self.page_index = 0
        self.library.hide()
        self.view.show()
        self.canvas.set_page(self.page)
        self.view.sync()
        self.canvas.setFocus()

    def go_home(self):
        self.canvas.commit_text()
        self.view.close_popovers()
        self.view.page_grid.hide()
        self.view.hide()
        self.library.sel = self.nb_index
        self.library.refresh()
        self.library.show()

    def new_notebook(self):
        name, ok = QInputDialog.getText(self, "New notebook", "Name:",
                                        text=f"Notebook {len(self.notebooks) + 1}")
        if not ok:
            return
        color = COVER_COLORS[len(self.notebooks) % len(COVER_COLORS)]
        self.notebooks.append(Notebook(name or "Notes", [Page()], color))
        self.library.sel = len(self.notebooks) - 1
        self.dirty = True
        self.library.refresh()

    def goto_page(self, idx):
        if not (0 <= idx < len(self.notebook.pages)):
            return
        self.canvas.commit_text()
        self.page_index = idx
        self.canvas.set_page(self.page)
        self.view.sync()

    def turn_page(self, delta):
        if self.view.isVisible():
            self.goto_page(self.page_index + delta)

    def turn_page_with_scroll(self, delta):
        if not self.view.isVisible():
            return
        target = self.page_index + delta
        if target < 0:
            return
        if target >= len(self.notebook.pages):
            self.add_page()
        else:
            self.goto_page(target)
        self._land_scroll(at_top=delta < 0)

    def _land_scroll(self, at_top):
        self.canvas.fill_view()
        page_h = PAGE_H * self.canvas.scale
        y = 0.0 if at_top else self.canvas.height() - page_h
        self.canvas.offset = QPointF(self.canvas.offset.x(), y)
        self.canvas._clamp()
        self.canvas.update()

    def add_page(self):
        if not self.view.isVisible():
            return
        self.canvas.commit_text()
        self.notebook.pages.insert(self.page_index + 1, Page(self.page.template))
        self.dirty = True
        self.goto_page(self.page_index + 1)
        if self.view.page_grid.isVisible():
            self.view.refresh_grid()

    def duplicate_page(self):
        self.canvas.commit_text()
        src = self.page
        self.notebook.pages.insert(
            self.page_index + 1,
            Page(src.template,
                 [Stroke.from_dict(s.to_dict()) for s in src.strokes],
                 [TextElement.from_dict(t.to_dict()) for t in src.texts]))
        self.dirty = True
        self.goto_page(self.page_index + 1)

    def delete_page(self):
        if len(self.notebook.pages) == 1:
            QMessageBox.information(self, APP_NAME, "A notebook needs at least one page.")
            return
        if QMessageBox.question(self, APP_NAME,
                                "Delete this page?") != QMessageBox.StandardButton.Yes:
            return
        self.canvas.commit_text()
        self.notebook.pages.pop(self.page_index)
        self.page_index = max(0, self.page_index - 1)
        self.dirty = True
        self.canvas.set_page(self.page)
        self.view.sync()

    def set_template(self, t):
        self.page.template = t
        self.canvas.refresh_cache()
        self.dirty = True

    def to_dict(self):
        return {"app": APP_NAME, "version": 2,
                "notebooks": [n.to_dict() for n in self.notebooks]}

    def load_dict(self, d):
        self.notebooks = [Notebook.from_dict(n) for n in d.get("notebooks", [])] \
            or [Notebook("Notes", [Page()])]
        self.nb_index = 0
        self.page_index = 0
        self.library.sel = 0
        self.library.refresh()
        if self.view.isVisible():
            self.canvas.set_page(self.page)
            self.view.sync()

    def _load_library(self):
        if os.path.exists(LIBRARY_PATH):
            try:
                with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                notebooks = [Notebook.from_dict(n) for n in data.get("notebooks", [])]
                if notebooks:
                    return notebooks
            except Exception:
                pass
        return [Notebook("Notes", [Page()], COVER_COLORS[0])]

    def _save_library(self):
        try:
            with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
        except Exception:
            pass

    def save_file(self):
        self.canvas.commit_text()
        if not self.file_path:
            return self.save_file_as()
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        self.dirty = False
        self.setWindowTitle(f"{APP_NAME} - {os.path.basename(self.file_path)}")

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save notebooks", "notebooks.inkpad",
                                              "InkPad files (*.inkpad)")
        if not path:
            return
        self.file_path = path
        self.save_file()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "InkPad files (*.inkpad)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.load_dict(json.load(f))
            self.file_path = path
            self.dirty = False
            self.setWindowTitle(f"{APP_NAME} - {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not open file:\n{exc}")

    def autosave(self):
        if self.dirty:
            self._save_library()

    def export_png(self):
        self.canvas.commit_text()
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG",
                                              f"page-{self.page_index + 1}.png",
                                              "PNG image (*.png)")
        if path:
            render_page(self.page).save(path, "PNG")

    def export_pdf(self, nb_index=None):
        self.canvas.commit_text()
        nb = self.notebooks[nb_index if nb_index is not None else self.nb_index]
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", f"{nb.name}.pdf",
                                              "PDF (*.pdf)")
        if not path:
            return
        write_pdf(nb, path)
        QMessageBox.information(self, APP_NAME,
                                f"Exported {len(nb.pages)} page(s) to\n{path}")

    def show_help(self):
        QMessageBox.information(self, APP_NAME, """
The page fills the whole window - the controls float on top of it.

Top left     library · all pages · undo · redo · fullscreen
Top right    pen (click again for tools + text tool) · colour · eraser · new page
Bottom       page arrows, page number, and the ... page menu

Text tool    Click 'Aa' in the pen menu to enable. Click anywhere on the paper to type.

Stylus       real pressure; the eraser end of the pen erases
Mouse        pressure is simulated from speed, so lines still taper

Scroll wheel / trackpad   pan around the page
Shift + wheel             pan sideways (for mice without a side wheel)
Keep scrolling past the top or bottom edge to move to the next/previous
page - past the last page, InkPad creates a new one instead of making
you erase to keep going.

Ctrl+Z / Ctrl+Y   undo / redo          Ctrl+N       new page
Ctrl+S / Ctrl+O   save / open          left, right  turn page
Ctrl+E / Ctrl+P   export PNG / PDF     F11          fullscreen
Ctrl+wheel        zoom in and out      Esc          back / exit fullscreen
Space+drag        pan when zoomed in
""".strip())

    def closeEvent(self, e):
        self.canvas.commit_text()
        if self.file_path and self.dirty:
            r = QMessageBox.question(
                self, APP_NAME, "Save changes to the open file before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Save:
                self.save_file()
            elif r == QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
        self._save_library()
        e.accept()


def write_pdf(nb: Notebook, path: str):
    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageOrientation(QPageLayout.Orientation.Landscape)
    writer.setResolution(150)
    painter = QPainter(writer)
    for i, pg in enumerate(nb.pages):
        if i:
            writer.newPage()
        vp = painter.viewport()
        painter.drawImage(QRectF(0, 0, vp.width(), vp.height()), render_page(pg))
    painter.end()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # App icon (Happy Tutors logo) - shown in the title bar, taskbar, and
    # Alt+Tab switcher. Falls back to Qt's default icon if the file is
    # ever missing, instead of crashing the app.
    if os.path.exists(APP_ICON_PATH):
        app_icon = QIcon(APP_ICON_PATH)
        app.setWindowIcon(app_icon)
    else:
        app_icon = QIcon()

    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI" if sys.platform.startswith("win") else "Sans Serif", 10))

    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #22262d;
            border: 1px solid #3c4149;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }
    """)

    win = InkPad()
    if not app_icon.isNull():
        win.setWindowIcon(app_icon)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()