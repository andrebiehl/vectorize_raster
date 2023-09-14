from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
import zipfile
from vectorize import vectorize_process
from flask import jsonify
from flask_mail import Mail, Message

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'tif', 'geotiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Flask-Mail with your email provider settings
app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@example.com'
app.config['MAIL_PASSWORD'] = 'your-password'

mail = Mail(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def zip_files():
    zipf = zipfile.ZipFile('vectorized.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(OUTPUT_FOLDER):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()

@app.route('/api/vectorize', methods=['POST'])
def vectorize_file():
    file = request.files['file']
    email = request.form['email']

    # Validate file type
    if file and allowed_file(file.filename):
        # Validate file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > 150 * 1024 * 1024:  # 150 MB
            return "File size exceeds 150 MB.", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Vectorize the file
        vectorize_process(filepath)
        zip_files()

        # Send the email
        zip_file_path = 'vectorized.zip'
        send_email(email, zip_file_path)

        return "Vectorization Complete. Check your email for the files.", 200
    else:
        return "File not allowed. Please upload a GeoTIFF file.", 400

def send_email(to_email, file_path):
    msg = Message("Vectorized Files", sender="your-email@example.com", recipients=[to_email])
    msg.body = "Here are the vectorized files you requested."
    with app.open_resource(file_path) as fp:
        msg.attach("vectorized.zip", "application/zip", fp.read())
    mail.send(msg)

@app.route('/download')
def download_file():
    return send_file('vectorized.zip',
                     mimetype='zip',
                     attachment_filename='vectorized.zip',
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
