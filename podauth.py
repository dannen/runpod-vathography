"""The pod's password, for requests to its ComfyUI and vathopy GUI.

On the pod, ComfyUI (port 8188, behind nginx) and the vathopy GUI (8770) both
require the user:password in VATHOPY_GUI_AUTH (see start.sh). The tools here
send it, as HTTP basic auth, on requests to the pod's proxy URLs
(https://<pod-id>-<port>.proxy.runpod.net) only, never to RunPod's API. It's
read from, in order:
  1. the VATHOPY_GUI_AUTH environment variable (user:password);
  2. the file in VATHOPY_GUI_AUTH_FILE, default ~/.api/runpod/vathopy-gui.auth.
"""

import base64
import os
import urllib.parse
from pathlib import Path

DEFAULT_FILE = "~/.api/runpod/vathopy-gui.auth"
HINT = ("the pod asked for its password: set VATHOPY_GUI_AUTH=user:password, or put user:password in "
        f"{DEFAULT_FILE} (or the file named by VATHOPY_GUI_AUTH_FILE)")

_cached = None


def credentials():
    """user:password, or None if none is configured."""
    global _cached
    if _cached is None:
        auth = os.environ.get("VATHOPY_GUI_AUTH", "").strip()
        if not auth:
            path = Path(os.environ.get("VATHOPY_GUI_AUTH_FILE") or DEFAULT_FILE).expanduser()
            try:
                auth = path.read_text().strip()
            except OSError:
                auth = ""
        _cached = auth if ":" in auth else ""
    return _cached or None


def is_pod_url(url):
    p = urllib.parse.urlsplit(url)
    return p.scheme == "https" and (p.hostname or "").endswith(".proxy.runpod.net")


def headers_for(url):
    """{"Authorization": "Basic ..."} for the pod's proxy URLs when a password is configured, else {}."""
    auth = credentials()
    if not auth or not is_pod_url(url):
        return {}
    return {"Authorization": "Basic " + base64.b64encode(auth.encode()).decode()}
