from typing import Union

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from apps.setup import setup_logging


setup_logging()


app = FastAPI()

mcp = FastApiMCP(
    app,
    name="trading-agent",
    description="Trading Agent MCP Server",
    describe_all_responses=True,
    describe_full_response_schema=True,
)

mcp.mount_http()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
