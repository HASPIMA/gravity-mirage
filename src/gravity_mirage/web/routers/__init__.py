from fastapi import APIRouter

from gravity_mirage.web.routers.delete import router as delete_router
from gravity_mirage.web.routers.exports import router as exports_router
from gravity_mirage.web.routers.img import router as img_router
from gravity_mirage.web.routers.preview import router as preview_router
from gravity_mirage.web.routers.uploads import router as uploads_router

api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(
    img_router,
)

api_router.include_router(
    preview_router,
)

api_router.include_router(
    uploads_router,
)

api_router.include_router(
    exports_router,
)

api_router.include_router(
    delete_router,
)

__all__ = ["api_router"]
