from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response
from app.car_plate_detection import detect_with_ensemble
from http import HTTPStatus
import json
from PIL import Image
import io
import cv2
import numpy as np
import os
from dotenv import load_dotenv

# import slowapi for API call limit
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# CORS to limit API calls from vercel or spicific domain
from fastapi.middleware.cors import CORSMiddleware

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app = FastAPI()

# limit API call
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

load_dotenv()
print("setting CORS domains: ", os.getenv("frontend_domain"))
# set CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("frontend_domain"),
    allow_methods=["POST"],  # only what you need
    allow_headers=["*"],
)


@app.post("/detect-plate/", dependencies=[])
@limiter.limit("5/minute")  # 10 requests per minute per IP
async def detect_car_plate_num(request: Request) -> Response:

    form = await request.form()
    file = form.get("file")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # read the file content into memory
        image_bytes = await file.read()

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        # convert bytes to PIL image
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # convert PIL Image to OpenCV format (BGR numpy array)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        text_list = detect_with_ensemble(image_cv)
        return Response(
            status_code=HTTPStatus.ACCEPTED, content=json.dumps({"message": text_list})
        )
    except Exception as e:
        return Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=json.dumps({"message": f"Failed to process image: {str(e)}"}),
        )
