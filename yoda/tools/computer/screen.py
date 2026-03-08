"""Screen capture and OCR tools."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from yoda.tools.registry import ToolPermission, tool

logger = logging.getLogger(__name__)


def register_screen_tools() -> None:
    """No-op — importing this module registers tools via decorators."""


@tool(
    name="screenshot",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["screen", "capture", "screenshot"],
    timeout=10.0,
)
async def screenshot(
    save_path: str = "",
    region: str = "",
    monitor: int = 0,
) -> str:
    """Take a screenshot of the screen or a region.

    Args:
        save_path: Path to save the screenshot (optional, returns base64 if empty).
        region: Region as 'x,y,width,height' (optional, full screen if empty).
        monitor: Monitor index for multi-monitor setups (default: 0).
    """
    try:
        import pyautogui
    except ImportError:
        return "Error: pyautogui not installed. Run: pip install pyautogui"

    try:
        if region:
            parts = [int(x.strip()) for x in region.split(",")]
            if len(parts) != 4:
                return "Error: region must be 'x,y,width,height'"
            img = pyautogui.screenshot(region=tuple(parts))
        else:
            img = pyautogui.screenshot()

        if save_path:
            p = Path(save_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(p))
            return f"Screenshot saved to {p} ({img.width}x{img.height})"
        else:
            # Return base64 encoded
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"Screenshot captured ({img.width}x{img.height}), base64 length: {len(b64)}"
    except Exception as e:
        return f"Screenshot failed: {e}"


@tool(
    name="screen_ocr",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["screen", "ocr", "text"],
    timeout=30.0,
)
async def screen_ocr(
    image_path: str = "",
    region: str = "",
    language: str = "eng",
) -> str:
    """Extract text from screen or image using OCR.

    Args:
        image_path: Path to image file (optional, captures screen if empty).
        region: Screen region as 'x,y,width,height' (if no image_path).
        language: OCR language (default: eng).
    """
    try:
        from PIL import Image
    except ImportError:
        return "Error: Pillow not installed. Run: pip install Pillow"

    # Get image
    if image_path:
        img = Image.open(Path(image_path).expanduser().resolve())
    else:
        try:
            import pyautogui
            if region:
                parts = [int(x.strip()) for x in region.split(",")]
                img = pyautogui.screenshot(region=tuple(parts))
            else:
                img = pyautogui.screenshot()
        except ImportError:
            return "Error: pyautogui not installed for screen capture"

    # Try pytesseract first
    try:
        import pytesseract
        text = pytesseract.image_to_string(img, lang=language)
        return text.strip() or "(no text detected)"
    except ImportError:
        pass

    # Fallback: try easyocr
    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader([language[:2]])
        img_array = np.array(img)
        results = reader.readtext(img_array)
        text = "\n".join(r[1] for r in results)
        return text.strip() or "(no text detected)"
    except ImportError:
        return (
            "Error: No OCR backend available. Install one of:\n"
            "  pip install pytesseract  (requires tesseract system package)\n"
            "  pip install easyocr"
        )


@tool(
    name="screen_info",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["screen", "info", "display"],
    timeout=5.0,
)
async def screen_info() -> str:
    """Get screen resolution and display information."""
    try:
        import pyautogui
        size = pyautogui.size()
        return f"Screen size: {size.width}x{size.height}"
    except ImportError:
        return "Error: pyautogui not installed"
    except Exception as e:
        return f"Could not get screen info: {e}"


@tool(
    name="locate_on_screen",
    permission=ToolPermission.COMPUTER,
    category="computer",
    tags=["screen", "find", "locate", "image"],
    timeout=15.0,
)
async def locate_on_screen(image_path: str, confidence: float = 0.8) -> str:
    """Find an image/icon on screen and return its location.

    Args:
        image_path: Path to the template image to find.
        confidence: Match confidence threshold (0.0-1.0).
    """
    try:
        import pyautogui
    except ImportError:
        return "Error: pyautogui not installed"

    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        return f"Image not found: {p}"

    try:
        location = pyautogui.locateOnScreen(str(p), confidence=confidence)
        if location:
            center = pyautogui.center(location)
            return (
                f"Found at: ({center.x}, {center.y})\n"
                f"Region: ({location.left}, {location.top}, {location.width}, {location.height})"
            )
        return "Image not found on screen"
    except Exception as e:
        return f"Locate failed: {e}"
