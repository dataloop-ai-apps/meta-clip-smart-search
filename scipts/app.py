import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter
import dtlpy as dl
import subprocess
import logging
import select
import os

logger = logging.getLogger('[CLEANUP]')
logging.basicConfig(level='INFO')

app = FastAPI()

origins = [
    "*",  # allow all
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=5463)
