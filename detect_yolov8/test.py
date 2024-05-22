from pathlib import Path

p = Path('data_img_crop')

for f in p.glob('**/*.jpg'):
    print(f)
