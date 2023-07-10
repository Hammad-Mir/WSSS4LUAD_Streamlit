import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
#import segmentation_models_pytorch as smp
from pytorch_grad_cam.utils.image import show_cam_on_image
from prep import preprocess_image, mask_img_to_mask, calculate_slice_bboxes, img_resize

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def predict_mask(model, img):
    
    image_tensor = preprocess_image(img,)
    
    pred_mask = model(image_tensor.to(device))
    pred_mask = torch.nn.functional.softmax(pred_mask, dim=1)
    pred_mask = np.transpose(pred_mask.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    return pred_mask

def load_model():
    #model = smp.DeepLabV3Plus(encoder_name='resnet50', classes=4, activation=None, encoder_weights=None, ).to(device)
    model = torch.load('models/deeplab.pth', map_location=device)
    model.load_state_dict(torch.load('.\models\deeplabv3plus_dJ_par_resnet50_01.pth', map_location=device))
    model.eval()

    return model

def predict(model, img_name):

    img_path  = 'images/img/' + str(img_name)
    mask_path = 'images/mask/' + str(img_name)
    bg_path   = 'images/background-mask/' + str(img_name)

    img     = cv2.imread(img_path)
    gt_mask = mask_img_to_mask(mask_path, bg_path)

    img, gt_mask, slice_boxes = img_resize(img, gt_mask, 224)
    pred = np.zeros(gt_mask.shape)

    for j in slice_boxes:
        pred[j[1]:j[3], j[0]:j[2]] = predict_mask(model, img[j[1]:j[3], j[0]:j[2]])
    
    pred = pred.round().astype('uint8')

    gt_t = (gt_mask[:, :, 0] + gt_mask[:, :, 1])
    gt_t[gt_t>1] = 1
    pred_t = (pred[:, :, 0] + pred[:, :, 1])
    pred_t[pred_t>1] = 1

    fig = plt.figure(figsize=(20, 20))
    
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(show_cam_on_image(img/255, gt_t, use_rgb=True))
    plt.title("Actual Tumor Tissue Area", fontsize = 40)
    
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(show_cam_on_image(img/255, gt_mask[:, :, 2], use_rgb=True))
    plt.title("Actual Normal Tissue Area", fontsize = 40)
    
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(show_cam_on_image(img/255, pred_t, use_rgb=True))
    plt.title("Predicted Tumor Tissue Area", fontsize = 40)
    
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(show_cam_on_image(img/255, pred[:, :, 2], use_rgb=True))
    plt.title("Predicted Normal Tissue Area", fontsize = 40)
    
    fig.tight_layout()
    #plt.savefig('temp.jpg')
    #plt.close(fig)
    return fig