"""
Unsplash 图片工具 — 根据关键词搜索高质量图片

Unsplash API 接口：
  GET https://api.unsplash.com/search/photos?query=xxx&client_id=xxx&per_page=5

返回结构（简化）：
  {
    "results": [
      {
        "id": "abc123",
        "urls": {
          "regular": "https://images.unsplash.com/photo-xxx?w=1080",  ← 我们用这个
          "small": "https://images.unsplash.com/photo-xxx?w=400",
          "thumb": "https://images.unsplash.com/photo-xxx?w=200"
        },
        "description": "A beautiful temple",
        "user": { "name": "Photographer Name" }
      }
    ]
  }

用途：给每个景点配一张图片，前端展示用。
"""

import httpx
from app.utils.config import settings

UNSPLASH_BASE_URL = "https://api.unsplash.com"


def search_photos(query: str, per_page: int = 5) -> list[dict]:
    """
    搜索图片

    Args:
        query: 搜索关键词，如 "故宫 Beijing" 或 "Great Wall China"
        per_page: 返回数量，默认 5 张

    Returns:
        图片列表，每个包含 url / description / photographer
        失败返回空列表
    """
    if not settings.unsplash_access_key:
        print("⚠️ UNSPLASH_ACCESS_KEY 未配置，跳过图片搜索")
        return []

    try:
        resp = httpx.get(
            f"{UNSPLASH_BASE_URL}/search/photos",
            params={
                "query": query,
                "per_page": per_page,
                "client_id": settings.unsplash_access_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        photos = []
        for item in data.get("results", []):
            photos.append({
                "url": item.get("urls", {}).get("regular", ""),
                "thumb": item.get("urls", {}).get("thumb", ""),
                "description": item.get("description") or item.get("alt_description", ""),
                "photographer": item.get("user", {}).get("name", ""),
            })

        return photos

    except Exception as e:
        print(f"❌ Unsplash 搜索异常: {e}")
        return []


def get_photo_url(query: str) -> str | None:
    """
    获取单张图片 URL（便捷方法）

    Args:
        query: 搜索关键词

    Returns:
        图片 URL，找不到返回 None
    """
    photos = search_photos(query, per_page=1)
    if photos:
        return photos[0].get("url")
    return None