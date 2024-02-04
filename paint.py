import pygame
from block import Block
import os
from test import predict
import torch
# from custom import VisionTransformer
from resnet import ResNet9

pygame.init()

WIDTH, HEIGHT = 28 * 20, 28 * 20

WIN = pygame.display.set_mode((WIDTH, HEIGHT))

data_dir = "./paint_dir"

blocks = [Block(20 * i, 20 * j, 20, 20) for i in range(28) for j in range(28)]

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = VisionTransformer(img_size=28, patch_size=4, in_chans=1, n_classes=10, embed_dim=8,
#                           depth=5, n_heads=2, mlp_ratio=4, qkv_bias=True, p=0, attn_p=0).to(device)

model = ResNet9(1, 10).to(device)

model.load_state_dict(torch.load("./resnet.pth"))


def redraw(win, blocks):
    win.fill((0, 0, 0))
    for block in blocks:
        block.draw(win)
    pygame.display.update()


running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_c]:
        for block in blocks:
            block.color = (255, 255, 255)

    if pygame.mouse.get_pressed()[0]:
        x, y = pygame.mouse.get_pos()
        block_x = y // 20
        block_y = x // 20

        blocks[block_y*28 + block_x].color = (0, 0, 0)

    file_name = "digit_{}.jpg".format(len(os.listdir(data_dir)))
    

    

    if keys[pygame.K_p]:
        img = data_dir + "/" + file_name
        pygame.image.save(WIN, img)
        predict(img, model, device)

    redraw(WIN, blocks)

pygame.quit()
