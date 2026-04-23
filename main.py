from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from recommend import get_product_names, recommend

app = FastAPI()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("frontend/index.html")


@app.get("/products/")
def list_products():
    return {"products": get_product_names()}


@app.get("/recommend/")
def get_recommend(product_name: str):
    try:
        results = recommend(product_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"recommendations": results}
