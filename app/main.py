"""
应用启动入口 — 相当于 Java 的 Application.java

这个文件做三件事：
1. 创建 FastAPI 实例
2. 配置 CORS（允许前端跨域请求）
3. 注册路由

运行方式：
    uvicorn app.main:app --reload
    或者直接：python -m app.main
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.config import settings
from app.api.routes import router

# 创建应用
app = FastAPI(
    title="智能旅行规划助手",
    description="基于 LangChain + LangGraph 的多智能体旅行规划系统",
    version="2.0.0",
    docs_url="/docs",      # Swagger UI 地址
    redoc_url="/redoc",    # ReDoc 地址
)

# 配置 CORS（让前端 localhost:5173 能访问后端 localhost:8000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "智能旅行规划助手",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


# 直接运行：python -m app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )