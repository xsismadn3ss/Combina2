import uvicorn

def main():
    uvicorn.run("src:app", port=8000, reload=True)


if __name__ == "__main__":
    main()
