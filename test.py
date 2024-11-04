import os
from model import CMTFusion
import torch
from torchvision.utils import save_image
import utils

def fuse_images(model, ir_path, vis_path, output_path, device):
    # IR 및 VIS 이미지 로드
    real_ir_imgs = utils.get_test_images(ir_path).to(device)
    real_rgb_imgs = utils.get_test_images(vis_path).to(device)

    # 모델을 사용하여 융합 이미지 생성
    with torch.no_grad():
        fused_img, _, _ = model(real_rgb_imgs, real_ir_imgs)
        
    # 융합된 이미지 저장
    save_image(fused_img, output_path, normalize=True)
    print(f"Saved fused image to: {output_path}")

def run_fusion(ir_dir, vis_dir, output_dir):
    # 모델 설정 및 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMTFusion().to(device)
    
    # 가중치 로드 및 module. 접두사 제거
    state_dict = torch.load('/content/drive/MyDrive/pretrained.pth', map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # module. 제거
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.eval()

    # 이미지 융합 수행
    for idx in range(1, 201):
        ir_path = os.path.join(ir_dir, f'IR{idx}.png')
        vis_path = os.path.join(vis_dir, f'VIS{idx}.png')
        output_path = os.path.join(output_dir, f'Fused_{idx}.png')
        
        # IR 및 VIS 이미지가 모두 있는 경우에만 융합 수행
        if os.path.exists(ir_path) and os.path.exists(vis_path):
            fuse_images(model, ir_path, vis_path, output_path, device)
        else:
            print(f"Image pair not found for index {idx}")

if __name__ == '__main__':
    # 구글 드라이브의 IR 및 VIS 이미지 경로와 융합 결과 저장 경로 설정
    ir_dir = '/content/drive/MyDrive/KAIST/IR'
    vis_dir = '/content/drive/MyDrive/KAIST/VIS'
    output_dir = '/content/drive/MyDrive/KAIST/Fused'
    
    # 융합 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 융합 실행
    run_fusion(ir_dir, vis_dir, output_dir)
