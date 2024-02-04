import pygame

class Block:
    def __init__(self, x, y, w, h, color=(255,255,255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def draw(self, win):
        rect = pygame.Rect(self.x, self.y, self.w, self.h)
        pygame.draw.rect(win, self.color, rect)
