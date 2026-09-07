from fastapi import FastAPI

from app.routers import extract, register_plate, search

app = FastAPI(title="Vehicle Re-ID API")

app.include_router(extract.router)
app.include_router(search.router)
app.include_router(register_plate.router)
