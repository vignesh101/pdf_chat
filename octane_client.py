from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


def _strip_html_to_text(s: str) -> str:
    import html as _html
    import re
    if not s:
        return ""
    try:
        s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
        s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
        s = re.sub(r"</?(?:br|p|div|h[1-6]|li|tr|td|th|ul|ol|table|section|article|header|footer)[^>]*>", "\n", s, flags=re.I)
        s = re.sub(r"<[^>]+>", " ", s)
        s = _html.unescape(s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    except Exception:
        return s


@dataclass
class OctaneAuthResult:
    ok: bool
    cookies: List[str]
    note: Optional[str]


class OctaneAPI:
    """Minimal ALM Octane helper.

    Attempts to authenticate using client credentials, capture cookies for the base domain,
    and provides thin helpers to fetch work items as plain text.
    """

    def __init__(self, base_url: str, client_id: str, client_secret: str, http_client: httpx.Client):
        self.base_url = (base_url or "").rstrip("/")
        self.client_id = client_id or ""
        self.client_secret = client_secret or ""
        self.http = http_client
        self._bearer: Optional[str] = None

    def _domain_cookies(self) -> List[str]:
        names: List[str] = []
        try:
            # httpx exposes a CookieJar; iterate to list cookie names present
            for cookie in self.http.cookies.jar:  # type: ignore[attr-defined]
                try:
                    names.append(cookie.name)
                except Exception:
                    continue
        except Exception:
            pass
        return sorted(list(set(names)))

    def login(self) -> OctaneAuthResult:
        """Try a few common auth endpoints to acquire cookies (or bearer).

        Returns cookie names captured for observability.
        """
        note: Optional[str] = None
        # Try legacy sign-in endpoint that may set session cookies
        try:
            url = f"{self.base_url}/authentication/sign_in"
            payload = {"client_id": self.client_id, "client_secret": self.client_secret}
            r = self.http.post(url, json=payload, headers={"Accept": "application/json"}, follow_redirects=True)
            if r.status_code < 400:
                return OctaneAuthResult(True, self._domain_cookies(), None)
        except Exception as e:
            note = f"sign_in failed: {str(e)[:100]}"

        # Try OAuth2 client_credentials
        oauth_candidates = [
            f"{self.base_url}/oauth/token",
            f"{self.base_url}/oauth2/token",
            f"{self.base_url}/authentication/oauth/token",
        ]
        body_pairs = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("ascii")
        for url in oauth_candidates:
            try:
                r = self.http.post(
                    url,
                    data=body_pairs,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Basic {basic}",
                    },
                    follow_redirects=True,
                )
                if r.status_code < 400:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    tok = data.get("access_token") if isinstance(data, dict) else None
                    if tok:
                        self._bearer = str(tok)
                        # Also attach Authorization for the client as a convenience
                        try:
                            self.http.headers.update({"Authorization": f"Bearer {self._bearer}"})
                        except Exception:
                            pass
                        return OctaneAuthResult(True, self._domain_cookies(), None)
            except Exception as e:
                note = f"oauth failed: {str(e)[:100]}"

        return OctaneAuthResult(False, self._domain_cookies(), note)

    def _h(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Accept": "application/json"}
        if self._bearer:
            h["Authorization"] = f"Bearer {self._bearer}"
        return h

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = self.http.get(url, params=params or {}, headers=self._h(), follow_redirects=True)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, None

    def fetch_sample_items(self, project_id: str, workspace_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Best-effort attempts to fetch Octane work items text.

        Tries a few likely endpoints. Returns a list of {key,title,text}.
        """
        out: List[Dict[str, Any]] = []
        candidates = [
            f"api/shared_spaces/{project_id}/workspaces/{workspace_id}/work_items",
            f"api/shared_spaces/{project_id}/workspaces/{workspace_id}/defects",
            f"api/shared_spaces/{project_id}/workspaces/{workspace_id}/stories",
            f"internal-api/shared_spaces/{project_id}/workspaces/{workspace_id}/work_items",
        ]
        params = {"limit": str(max(1, min(50, int(limit))))}
        for path in candidates:
            try:
                code, data = self._get_json(path, params=params)
                if code >= 400 or not data:
                    continue
                items = []
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    items = data["data"]
                elif isinstance(data, list):
                    items = data
                for it in items:
                    try:
                        iid = str(it.get("id") or it.get("uid") or it.get("_id") or "")
                        name = (it.get("name") or it.get("title") or f"Item {iid}")
                        desc = it.get("description_html") or it.get("description") or it.get("content") or ""
                        text = _strip_html_to_text(desc) if isinstance(desc, str) else (json.dumps(desc) if desc else "")
                        key = f"octane:{iid}" if iid else f"octane:{name}"
                        out.append({"key": key, "title": name, "text": text})
                    except Exception:
                        continue
                if out:
                    break
            except Exception:
                continue
        return out

