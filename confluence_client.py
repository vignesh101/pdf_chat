from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx


def _strip_html_to_text(s: str) -> str:
    # Basic HTML to text: remove scripts/styles, strip tags, unescape
    if not s:
        return ""
    try:
        s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
        s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
        s = re.sub(r"</?(?:br|p|div|h[1-6]|li|tr|td|th|ul|ol|table|section|article|header|footer)[^>]*>", "\n", s, flags=re.I)
        s = re.sub(r"<[^>]+>", " ", s)
        s = html.unescape(s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    except Exception:
        return s


class ConfluenceAPI:
    """Minimal Confluence REST API helper for searching and fetching page content.

    Supports Atlassian Cloud and Server/DC patterns by trying both /wiki/rest/api and /rest/api.
    Authentication:
      - If access_token contains a colon ("user:token"), it is sent as Basic auth.
      - Otherwise, it's sent as Bearer token.
    """

    def __init__(self, base_url: str, access_token: str, http_client: httpx.Client):
        # Accept base URL with or without trailing '/wiki'
        self.base_url = (base_url or "").rstrip("/")
        self.token = access_token or ""
        self.http = http_client

    def _auth_headers(self) -> Dict[str, str]:
        if ":" in (self.token or ""):
            import base64
            b = base64.b64encode(self.token.encode("utf-8")).decode("ascii")
            return {"Authorization": f"Basic {b}"}
        else:
            return {"Authorization": f"Bearer {self.token}"}

    def _url_candidates(self, path: str) -> List[str]:
        base = self.base_url.rstrip("/")
        p = path.lstrip("/")
        # If base already ends with /wiki, avoid duplicating it
        if base.endswith("/wiki"):
            root = base[:-5]  # strip trailing '/wiki'
            return [
                f"{base}/rest/api/{p}",      # https://host/wiki/rest/api/...
                f"{root}/rest/api/{p}",      # https://host/rest/api/... (Server/DC cases)
            ]
        # Otherwise try Cloud first, then Server/DC
        return [
            f"{base}/wiki/rest/api/{p}",
            f"{base}/rest/api/{p}",
        ]

    def _absolute_url(self, url: str | None) -> str:
        """Return absolute URL for Confluence page links (handles relative '/wiki/...' cases)."""
        u = (url or '').strip()
        if not u:
            return ""
        if u.startswith("http://") or u.startswith("https://"):
            return u
        # Build from site root
        base = self.base_url.rstrip("/")
        # If we were given '/wiki/...', keep it; otherwise just join
        if u.startswith("/"):
            return f"{base}{u}"
        return f"{base}/{u}"

    def search_pages(self, query: str, spaces: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        limit = max(1, min(25, int(limit)))
        # Build CQL: type = page AND text ~ "..." [AND space in ("A","B")]
        q = query.replace('"', '\\"')
        cql = f'type = page AND text ~ "{q}"'
        if spaces:
            # quote each space key
            keys = ",".join([f'"{s.strip()}"' for s in spaces if s and s.strip()])
            if keys:
                cql += f" AND space in ({keys})"
        params = {"cql": cql, "limit": str(limit)}
        headers = {"Accept": "application/json"}
        headers.update(self._auth_headers())

        # Try search endpoints
        for url in self._url_candidates("search"):
            try:
                r = self.http.get(url, params=params, headers=headers, follow_redirects=True)
                if r.status_code >= 400:
                    continue
                data = r.json()
                results = data.get("results") or []
                out: List[Dict[str, Any]] = []
                for it in results:
                    try:
                        content = it.get("content") or {}
                        cid = str(content.get("id") or "")
                        title = content.get("title") or it.get("title") or ""
                        page_url = self._absolute_url(it.get("url") or "")
                        out.append({"id": cid, "title": title, "url": page_url})
                    except Exception:
                        continue
                if out:
                    return out
            except Exception:
                continue
        return []

    def get_page_content_text(self, page_id: str) -> Tuple[str, Optional[str]]:
        """Return (text, title) for a Confluence page id."""
        headers = {"Accept": "application/json"}
        headers.update(self._auth_headers())
        params = {"expand": "body.view,body.export_view,body.storage"}
        for url in self._url_candidates(f"content/{page_id}"):
            try:
                r = self.http.get(url, params=params, headers=headers, follow_redirects=True)
                if r.status_code >= 400:
                    continue
                data = r.json()
                title = (data.get("title") or "").strip() or None
                body = data.get("body") or {}
                html_view = None
                try:
                    html_view = (body.get("view") or {}).get("value")
                except Exception:
                    html_view = None
                if not html_view:
                    try:
                        html_view = (body.get("export_view") or {}).get("value")
                    except Exception:
                        html_view = None
                if not html_view:
                    try:
                        html_view = (body.get("storage") or {}).get("value")
                    except Exception:
                        html_view = None
                if isinstance(html_view, str) and html_view.strip():
                    return _strip_html_to_text(html_view), title
                # Last resort: if body missing, return empty
                return "", title
            except Exception:
                continue
        return "", None
