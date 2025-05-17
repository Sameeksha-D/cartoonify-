import os
import cv2
import numpy as np
from pillow_heif import HeifImage
from PIL import Image
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = "static/cartoon_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  Support multiple image formats 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "heic"}

def allowed_file(filename):
    """Check if uploaded file has an allowed image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

generated_images = []  # Store last three processed images

def convert_heic_to_jpg(heic_path):
    """Convert HEIC image to JPEG format using pillow-heif."""
    heif_img = HeifImage.open(heic_path)
    img = heif_img.to_pil()

    jpg_path = heic_path.replace(".heic", ".jpg")  # Convert HEIC filename to JPG
    img.save(jpg_path, "JPEG")
    
    return jpg_path

def cartoonify_image(image_path, image_type):
    """Apply customized cartoon effect based on selected image type."""
    img = cv2.imread(image_path)
    original_size = img.shape[:2]


    img = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔹 Adjust cartoonification based on image type
    if image_type in ["human", "nature"]:
        blur = cv2.medianBlur(gray, 7)  
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 200, 200)
    elif image_type == "object":  
        blur = cv2.medianBlur(gray, 3) 
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)  
        color = cv2.bilateralFilter(img, 9, 230, 230)  
    elif image_type == "animal":
        blur = cv2.medianBlur(gray, 3)  
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        color = cv2.bilateralFilter(img, 9, 180, 180)  
    elif image_type == "black":
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  
        blur = cv2.medianBlur(gray, 3)
        edges = cv2.Canny(blur, 30, 80) 
        color = cv2.bilateralFilter(img, 12, 350, 350)

    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon_resized = cv2.resize(cartoon, (original_size[1], original_size[0]))

    return cartoon_resized

@app.route("/", methods=["GET", "POST"])
def index():
    """Handle file upload and image processing."""
    if request.method == "POST":
        file = request.files["image"]
        image_type = request.form["image_type"]  

        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            if file.filename.lower().endswith(".heic"):
                filepath = convert_heic_to_jpg(filepath)

            cartoon_image = cartoonify_image(filepath, image_type)
            cartoon_filepath = os.path.join(UPLOAD_FOLDER, "cartoon_" + os.path.basename(filepath))
            cv2.imwrite(cartoon_filepath, cartoon_image)

            generated_images.insert(0, {
                "original": filepath.replace("\\", "/"),
                "cartoon": cartoon_filepath.replace("\\", "/")
            })

            if len(generated_images) > 3:
                old_images = generated_images.pop()
                os.remove(old_images["original"])
                os.remove(old_images["cartoon"])

    return render_template("index.html", images=generated_images)

@app.route("/download/<path:image_path>")
def download_file(image_path):
    """Allow users to download their cartoonified images."""
    return send_file(image_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)