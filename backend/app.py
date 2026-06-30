from fastapi import FastAPI, HTTPException, Request
from backend.car_plate_detection_tesseract import detect_with_yolo
from http import HTTPStatus
from PIL import Image
import io
import cv2
import numpy as np
import gc
import ctypes
import base64

# import slowapi for API call limit
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# CORS to limit API calls from vercel or spicific domain
from fastapi.middleware.cors import CORSMiddleware

import torch

torch.set_num_threads(1)

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB

app = FastAPI()

# limit API call
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# set CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://portfolio-bekzat-kb.vercel.app"
    ],  # os.getenv("frontend_domain"),
    allow_methods=["POST"],  # only what you need
    allow_headers=["*"],
)


def cleanup_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def resize_if_large(image_cv, max_dim=1280):
    h, w = image_cv.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_cv = cv2.resize(image_cv, (int(w * scale), int(h * scale)))
    return image_cv


@app.post("/detect-plate/", dependencies=[])
@limiter.limit("5/minute")  # 10 requests per minute per IP
async def detect_car_plate_num(request: Request):

    form = await request.form()
    file = form.get("file")

    if not file or not file.content_type.startswith("image/"):
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

        del image_bytes, image_pil, file

        image_cv = resize_if_large(image_cv)
        text_list, confidence_list, plate_crop_list = detect_with_yolo(image_cv)

        plate_images = []
        for plate_crop in plate_crop_list:
            success, encoded_image = cv2.imencode(".jpg", plate_crop)
            if not success:
                continue
            plate_images.append(
                "data:image/jpeg;base64,"
                + base64.b64encode(encoded_image.tobytes()).decode("utf-8")
            )

        return {
            "status_code": HTTPStatus.ACCEPTED,
            "plate_number": text_list,
            "confidence": confidence_list,
            "plate_images": plate_images,
        }

    except Exception as e:
        return {
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": f"Failed to process image: {str(e)}",
        }
    finally:
        cleanup_memory()
