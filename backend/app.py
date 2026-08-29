from backend.main import app

if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)
