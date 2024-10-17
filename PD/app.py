from flask import Flask, request, render_template, url_for, redirect # type: ignore
import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import os

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'PD/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Định nghĩa ConvBlock (block tích chập)
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)

# Định nghĩa mô hình ResNet-9
class ResNet9(torch.nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        # Sử dụng AdaptiveAvgPool2d để luôn có kích thước đầu ra cố định (1x1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.res2 = torch.nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)  # Đảm bảo đầu ra là (512x1x1)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Định nghĩa các class (tên bệnh cây)
disease_classes = [
    'Apple_scab', 'Apple_black_rot', 'Cedar_apple_rust', 'Healthy_apple',
    'Blueberry_healthy', 'Cherry_powdery_mildew', 'Cherry_healthy',
    'Corn_(maize)_gray_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 
    'Corn_healthy', 'Grape_black_rot', 'Grape_esca', 'Grape_leaf_blight', 'Grape_healthy',
    'Orange_haunglongbing', 'Peach_bacterial_spot', 'Peach_healthy',
    'Pepper_bell_bacterial_spot', 'Pepper_healthy', 'Potato_early_blight',
    'Potato_late_blight', 'Potato_healthy', 'Soybean_healthy',
    'Squash_powdery_mildew', 'Strawberry_healthy', 'Tomato_bacterial_spot',
    'Tomato_early_blight', 'Tomato_late_blight', 'Tomato_leaf_mold',
    'Tomato_septoria_leaf_spot', 'Tomato_spider_mites', 'Tomato_target_spot',
    'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus'
]


model = ResNet9(in_channels=3, num_diseases=len(disease_classes))
model = torch.load('PD\model\plant-disease-model-complete.pth', map_location=torch.device('cpu'))
model.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(image):
    if image.size[0] < 224 or image.size[1] < 224:
        raise ValueError("Image size is too small. Minimum size is 224x224.")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Dự đoán bệnh cây
def predict(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return "Invalid image"

    input_tensor = transform_image(image)
    print(f"Input Tensor Shape: {input_tensor.shape}")  # Debugging

    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return disease_classes[predicted_idx.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file selected", 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                result = predict(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')
                return redirect(url_for('result', result=result, image_url=image_url))
            except Exception as e:
                print(f"Prediction error: {e}")
                return f"Error during prediction: {e}", 500
    return render_template('demo/upload.html')

@app.route('/result')
def result():
    result = request.args.get('result')
    image_url = request.args.get('image_url')
    return render_template('demo/result.html', result=result, image_url=image_url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)