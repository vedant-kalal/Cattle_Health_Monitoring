import os
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

class CattleDiseaseModel:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        # Avoid torch.hub (which may attempt network calls). Use torchvision API directly.
        self.model = resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 3)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.class_names = ['healthy', 'lsd', 'mastitis']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            disease_name = self.class_names[predicted_class]
        return disease_name

    def predict_folder(self, folder_path):
        import glob
        results = []
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        for img_path in image_paths:
            try:
                disease = self.predict(img_path)
                results.append((os.path.basename(img_path), disease))
            except Exception as e:
                results.append((os.path.basename(img_path), f'Error: {e}'))
        return results
