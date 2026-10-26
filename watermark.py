"""Attribution watermark applied to every figure, frame and video this project writes.

Change TEXT here to change it everywhere. Keep it to plain ASCII without quotes or colons, because
the same string is passed to ffmpeg's drawtext filter.

This is attribution, not protection: anyone who edits this file can remove it.
"""

import cv2

TEXT = "(c) Kamal Kumar Behera | github.com/STAKXX002/SVP_Project"


def save_figure(fig, path, **kwargs):
    """Add a faint diagonal mark and a footer to a matplotlib figure, then save it."""
    fig.text(0.5, 0.5, TEXT, ha="center", va="center", rotation=25, fontsize=22,
             color="gray", alpha=0.28, transform=fig.transFigure, zorder=1000)
    fig.text(0.995, 0.004, TEXT, ha="right", va="bottom", fontsize=7, color="dimgray",
             transform=fig.transFigure, zorder=1000)
    fig.savefig(path, **kwargs)


def mark_frame(frame):
    """Draw the mark on a BGR frame (in place) and return it."""
    h = frame.shape[0]
    scale = max(0.4, h / 900.0)
    thick = 1 if h < 700 else 2
    pos = (8, h - 10)
    cv2.putText(frame, TEXT, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, TEXT, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return frame


def ffmpeg_drawtext():
    """drawtext filter string for ffmpeg -vf chains."""
    return (f"drawtext=text='{TEXT}':x=10:y=h-th-10:fontsize=h/28:fontcolor=white@0.8:"
            f"box=1:boxcolor=black@0.45:boxborderw=4")
