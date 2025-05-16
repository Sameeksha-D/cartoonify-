import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = "static/cartoon_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

generated_images = []  # Store last three processed images

def cartoonify_image(image_path):
    img = cv2.imread(image_path)
    original_size = img.shape[:2]  # Get (height, width) of original

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Resize cartoonified image to match original
    cartoon_resized = cv2.resize(cartoon, (original_size[1], original_size[0]))

    return cartoon_resized

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        
        if file and file.mimetype.startswith("image"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            cartoon_image = cartoonify_image(filepath)
            cartoon_filepath = os.path.join(UPLOAD_FOLDER, "cartoon_" + file.filename)
            cv2.imwrite(cartoon_filepath, cartoon_image)

            generated_images.insert(0, {  # Add new images at the top
                "original": filepath.replace("\\", "/"),
                "cartoon": cartoon_filepath.replace("\\", "/")
            })

            if len(generated_images) > 3:
                old_images = generated_images.pop()  # Remove oldest image
                os.remove(old_images["original"])
                os.remove(old_images["cartoon"])

    return render_template("index.html", images=generated_images)

@app.route("/download/<path:image_path>")
def download_file(image_path):
    return send_file(image_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)