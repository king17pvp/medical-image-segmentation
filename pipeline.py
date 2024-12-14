import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from models.classification_models.ResNet import *
from models.segmentation_models.ResnetUnet import *

class Pipeline:
    def __init__(self, img_size=256):
        self.transform = self._get_transforms(img_size)
        self.classification_model, self.segmentation_model, self.lungs_model = self._load_models()
        self.class_names = ['COVID', 'Non-COVID', 'Healthy']
        
    def _get_transforms(self, img_size):
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _load_models(self):
        classification_model = resnet_model
        classification_model.load_state_dict(torch.load('weights/classification_models/resnet50.pt', map_location=torch.device('cpu')))
        classification_model.eval()

        segmentation_model = ResNetUnet()
        checkpoint = torch.load('weights/segmentation_models/ResNetUnet_best.pt', map_location=torch.device('cpu'))
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        segmentation_model.eval()

        lungs_model = ResNetUnet()
        checkpoint = torch.load('weights/segmentation_models/full_lungs_resnet.pt')
        lungs_model.load_state_dict(checkpoint['model_state_dict'])
        lungs_model.eval()

        return classification_model, segmentation_model, lungs_model
    
    def process_image(self, image, overlay_opacity=0.5):
        if image is None:
            return None, None, None, None
   
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0)
        
        with torch.inference_mode():
            outputs = self.classification_model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100

        prediction = self.class_names[pred_class]
        
        if prediction == 'COVID':
            with torch.inference_mode():
                # [COVID MASK]
                covid_output = self.segmentation_model(input_tensor)
                covid_output = torch.sigmoid(covid_output)
                covid_output = covid_output.squeeze().cpu().numpy()
                covid_binary_mask = (covid_output > 0.5).astype(np.uint8) * 255
                covid_mask_resized = cv2.resize(covid_binary_mask, (image.shape[1], image.shape[0]))
                
                covid_overlay = np.zeros_like(image)
                covid_overlay[covid_mask_resized > 0] = [255, 0, 0]  
                
                # [LUNGS MASK]
                lung_output = self.lungs_model(input_tensor)
                lung_output = torch.sigmoid(lung_output)
                lung_output = lung_output.squeeze().cpu().numpy()
                lung_binary_mask = (lung_output > 0.5).astype(np.uint8) * 255
                lung_mask_resized = cv2.resize(lung_binary_mask, (image.shape[1], image.shape[0]))
                
                lung_overlay = np.zeros_like(image)
                lung_overlay[lung_mask_resized > 0] = [0, 255, 0] 
                
                # [REGION CALCULATION]
                covid_area = np.sum(covid_mask_resized > 0)
                lung_area = np.sum(lung_mask_resized > 0)
                infected_percentage = (covid_area / lung_area) * 100

                combined_overlay = cv2.addWeighted(covid_overlay, overlay_opacity, lung_overlay, overlay_opacity, 0)
                
                blended = cv2.addWeighted(image, 1, combined_overlay, 1, 0)

                analysis_text = (
                    f"COVID-19 Detection Results:\n"
                    f"• Infection severity of {infected_percentage:.2f}%\n"
                    f"• Yellow and red overlay indicates areas of potential COVID-19 infection\n"
                    f"• Recommended: Seek immediate medical attention"
                )
                return prediction, confidence, blended, analysis_text
                
        elif prediction == 'Non-COVID':
            analysis_text = (
                f"Non-COVID Lung Condition Detected:\n"
                f"• Confidence: {confidence:.1f}%\n"
                f"• Other lung abnormalities as pneumonia or lungs enlargement should be considered for further treatment\n"
                f"• Recommended: Consult with healthcare provider for proper diagnosis"
            )
            return prediction, confidence, None, analysis_text
            
        else:  
            analysis_text = (
                f"Healthy Lung Scan Results:\n"
                f"• Confidence: {confidence:.1f}%\n"
                f"• No significant abnormalities detected :)\n"
                f"• Regular check-ups is recommended"
            )
            return prediction, confidence, None, analysis_text