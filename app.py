import os
import cv2
import numpy as np
from pillow_heif import HeifImage
from PIL import Image
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = "static/cartoon_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Support multiple image formats including HEIC
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

def cartoonify_image(image_path):
    """Apply a sharper cartoon effect for animal & nature images."""
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Get (height, width)

    # ✅ Boost contrast to retain sharp details
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

    # ✅ Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)

    # ✅ Stronger edge detection for more defined outlines
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)

    # ✅ Preserve image details with better bilateral filtering
    color = cv2.bilateralFilter(img, 9, 250, 250)

    # ✅ Overlay edges on the filtered image for a crisp cartoon effect
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # ✅ Maintain original size
    cartoon_resized = cv2.resize(cartoon, (original_size[1], original_size[0]))

    return cartoon_resized

@app.route("/", methods=["GET", "POST"])
def index():
    """Handle file upload and image processing."""
    if request.method == "POST":
        file = request.files["image"]

        if file and allowed_file(file.filename):  # ✅ Validate image format
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # ✅ Convert HEIC to JPG before processing
            if file.filename.lower().endswith(".heic"):
                filepath = convert_heic_to_jpg(filepath)

            cartoon_image = cartoonify_image(filepath)
            cartoon_filepath = os.path.join(UPLOAD_FOLDER, "cartoon_" + os.path.basename(filepath))
            cv2.imwrite(cartoon_filepath, cartoon_image)

            generated_images.insert(0, {
                "original": filepath.replace("\\", "/"),
                "cartoon": cartoon_filepath.replace("\\", "/")
            })

            # ✅ Keep only the last three images for cleanup
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