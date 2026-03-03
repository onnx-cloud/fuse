from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse

from .models import (
    LintRequest,
    LintResponse,
    CompileRequest,
    CompileResponse,
    DecompileRequest,
    DecompileResponse,
)
from .handlers import lint_handler, compile_handler, decompile_handler

app = FastAPI(title="Fuse Server", docs_url="/docs", redoc_url="/redoc")
router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.post("/lint", response_model=LintResponse)
async def lint(req: LintRequest):
    return lint_handler(req)


@router.post("/compile", response_model=CompileResponse)
async def compile(req: CompileRequest):
    # Always return 200 with success flag in body; handler encodes errors in body
    return compile_handler(req)


@router.post("/decompile", response_model=DecompileResponse)
async def decompile(req: DecompileRequest):
    res = decompile_handler(req)
    # If decompile failed due to validation, return 400
    if not res.success and res.errors and any("missing" in (e.get("message","")) for e in res.errors):
        return JSONResponse(status_code=400, content=res.dict())
    return res


# provide a simple root health endpoint
@app.get("/health")
async def root_health():
    return {"status": "ok", "version": "0.1.0"}


app.include_router(router)
