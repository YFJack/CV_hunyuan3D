import os
import sys
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocess.image_preprocess import preprocess_image_for_model

def test_preprocess():
    # Create a dummy image (white circle on black background)
    img = Image.new('RGB', (500, 500), (0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 100, 400, 400), fill=(255, 255, 255))
    
    print("Testing preprocessing...")
    try:
        result = preprocess_image_for_model(img)
        processed_img = result['image']
        print(f"Success! Output shape: {processed_img.size}")
        
        # Save for inspection
        os.makedirs("outputs", exist_ok=True)
        processed_img.save("outputs/test_preprocess.png")
        print("Saved to outputs/test_preprocess.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocess()
