from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse

from gravity_mirage.web.constants import CHUNK_SIZE
from gravity_mirage.web.utils.files import (
    allocate_image_path,
    resolve_uploaded_file,
    sanitize_extension,
)

router = APIRouter(
    prefix="/uploads",
    tags=["uploads"],
)


@router.post("/")
async def upload_new_image(file: Annotated[UploadFile, File()]) -> RedirectResponse:
    """Persist an uploaded file and redirect back to the UI."""
    if file is None or not file.filename:
        return RedirectResponse(
            "/",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    ext = sanitize_extension(Path(file.filename).suffix)
    dest = allocate_image_path(ext)

    with dest.open("wb") as buffer:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            buffer.write(chunk)
    await file.close()
    return RedirectResponse(
        "/",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/{filename:path}")
async def get_uploaded_asset(filename: str) -> FileResponse:
    """Serve original uploaded assets."""
    path = resolve_uploaded_file(filename)
    return FileResponse(path)


__all__ = ["router"]
