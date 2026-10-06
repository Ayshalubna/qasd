"""Run the routing API and demo locally:  python -m scripts.serve  ->  http://localhost:8000"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("qasd.api:api", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
