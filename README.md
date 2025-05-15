from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import font_manager

# 设置中文字体（需要系统中存在该字体）
font_path = font_manager.findfont(font_manager.FontProperties(family='simhei'))
font = ImageFont.truetype(font_path, 60)

# 生成动态帧
frames = []
for angle in np.arange(0, 360, 10):  # 0到360度旋转
    # 创建画布
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 动态绘制文字
    text = "中国科学院"
    w, h = draw.textsize(text, font=font)
    position = ((400-w)/2, (200-h)/2)
    
    # 创建旋转文本
    txt = Image.new('RGBA', (w, h), (255,255,255,0))
    d = ImageDraw.Draw(txt)
    d.text((0,0), text, font=font, fill=(0,120,255))
    rotated = txt.rotate(angle, expand=1)
    
    # 合成图像
    img.paste(rotated, (int(position[0]), int(position[1])), rotated)
    
    frames.append(img)

# 保存为GIF
frames[0].save(
    'cas_animation.gif',
    save_all=True,
    append_images=frames[1:],
    optimize=True,
    duration=100,
    loop=0
)

print("动态GIF已生成：cas_animation.gif")
