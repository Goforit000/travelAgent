from __future__ import annotations

from typing import overload

from primp import Client
import httpx as _httpx

from .integrations.base import Integration
from .parser import MetaList, parse
from .querying import Query

URL = "https://www.google.com/travel/flights"
SEARCH_URL = "https://www.google.com/travel/flights/search"

CONSENT_COOKIE = (
    "CONSENT=YES+srp.gws-20240523-0-RC2.en+FX+999; "
    "SOCS=CAESEwgCEgk; "
    "AEC=AakniGNSO-...; "
    "NID=511=test"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
}


@overload
def get_flights(q: str, /, *, proxy: str | None = None) -> MetaList: ...


@overload
def get_flights(q: Query, /, *, proxy: str | None = None) -> MetaList: ...


def get_flights(
    q: Query | str,
    /,
    *,
    proxy: str | None = None,
    integration: Integration | None = None,
) -> MetaList:
    html = fetch_flights_html(q, proxy=proxy, integration=integration)
    return parse(html)


def fetch_flights_html(
    q: Query | str,
    /,
    *,
    proxy: str | None = None,
    integration: Integration | None = None,
) -> str:
    if integration is not None:
        return integration.fetch_html(q)

    if isinstance(q, Query):
        target_url = q.url()
    else:
        target_url = f"{SEARCH_URL}?q={q}"

    # 策略 A: httpx — cookie 在重定向中自动保留，先访问 google.com 设基础 cookie
    try:
        with _httpx.Client(timeout=15, follow_redirects=True, verify=False) as hc:
            hc.get("https://www.google.com/", headers={**HEADERS, "Cookie": CONSENT_COOKIE})
            resp = hc.get(target_url, headers=HEADERS)
            final_url = str(resp.url)
            final_text = resp.text
            if "consent.google.com" not in final_url:
                return final_text
    except Exception:
        pass

    # 策略 B: primp — 带 Chrome 指纹伪装，用 header 传 CONSENT cookie
    try:
        client = Client(
            impersonate="chrome_145",
            impersonate_os="macos",
            referer=True,
            proxy=proxy,
            cookie_store=True,
        )

        if isinstance(q, Query):
            params = q.params()

        res = client.get(SEARCH_URL, params=params, headers={"Cookie": CONSENT_COOKIE})
        if res is None:
            raise RuntimeError("primp returned None response")

        final_url = str(res.url)
        final_text = res.text
        if "consent.google.com" not in final_url:
            return final_text
    except Exception:
        pass

    raise RuntimeError(
        "[fast_flights] 你的 IP 被 Google 识别为 GDPR 区域（Europe/UK），"
        "所有的请求都被重定向到 cookie 同意页面。\n"
        "解决方法：\n"
        "  1. 使用代理/VPN 切换到非欧洲地区（美国、香港、日本、新加坡等）\n"
        "  2. 在 query_flights() 中尝试使用 BrightData 集成\n"
        "  3. 如果本地有代理，设置环境变量 HTTPS_PROXY 后重试"
    )
