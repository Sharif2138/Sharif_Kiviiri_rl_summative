import pygame
import numpy as np


class DriverFatigueRenderer:
    def __init__(self, width=800, height=420):
        pygame.init()
        pygame.display.set_caption("AI Fleet Road Safety Monitor")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_main = pygame.font.SysFont("Arial", 18)
        self.font_status = pygame.font.SysFont("Arial", 22, bold=True)
        self.clock = pygame.time.Clock()

    def render(self, state, action, current_step, reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        perclos, yawn, head_pose, drive_time = state

        self.screen.fill((20, 24, 33))

        header_rect = pygame.Rect(15, 15, 770, 50)
        pygame.draw.rect(self.screen, (35, 42, 58),
                         header_rect, border_radius=8)
        title_text = self.font_title.render(
            "COMMERCIAL FLEET DRIVER MONITORING SYSTEM", True, (255, 255, 255))
        self.screen.blit(title_text, (35, 25))

        if perclos < 0.35:
            status_color = (46, 204, 113)
            status_str = "STATUS: DRIVER ALERT"
        elif perclos < 0.70:
            status_color = (241, 196, 15)
            status_str = "STATUS: MILD FATIGUE DETECTED"
        else:
            status_color = (231, 76, 60)
            status_str = "STATUS: CRITICAL DROWSINESS WARNING!"

        status_rect = pygame.Rect(15, 80, 770, 45)
        pygame.draw.rect(self.screen, status_color,
                         status_rect, border_radius=8)
        status_text = self.font_status.render(
            status_str, True, (10, 10, 10) if perclos < 0.7 else (255, 255, 255))
        self.screen.blit(status_text, (35, 90))

        telemetry_rect = pygame.Rect(15, 140, 480, 200)
        pygame.draw.rect(self.screen, (30, 36, 50),
                         telemetry_rect, border_radius=8)

        telemetry_items = [
            f"Shift Progress: Step {current_step}/100",
            f"Continuous Driving Time: {drive_time:.1f} Mins",
            f"PERCLOS (Eye Closure Rate): {perclos*100:.1f}%",
            f"Yawn Frequency: {yawn:.1f} / min",
            f"Head Pitch Angle: {head_pose:.1f}°",
            f"Step Reward: {reward:.2f}"
        ]

        for idx, line in enumerate(telemetry_items):
            txt = self.font_main.render(line, True, (210, 220, 235))
            self.screen.blit(txt, (30, 155 + (idx * 28)))

        action_rect = pygame.Rect(510, 140, 275, 200)
        pygame.draw.rect(self.screen, (30, 36, 50),
                         action_rect, border_radius=8)

        action_labels = [
            ("0: MONITORING", (120, 140, 160)),
            ("1: MILD DRIVER VIBRATION", (241, 196, 15)),
            ("2: IN-BUS RED LIGHTS AND MILD ALARM", (231, 76, 60)),
            ("3: MANAGER CRITICAL ALARM", (155, 89, 182))
        ]

        act_title = self.font_title.render("AI ACTION", True, (255, 255, 255))
        self.screen.blit(act_title, (530, 155))

        act_label, act_color = action_labels[action] if action is not None else action_labels[0]
        act_box = pygame.Rect(525, 200, 245, 60)
        pygame.draw.rect(self.screen, act_color, act_box, border_radius=6)

        act_txt = self.font_status.render(
            act_label, True, (255, 255, 255) if action != 1 else (10, 10, 10))
        self.screen.blit(act_txt, (535, 215))

        footer_text = self.font_main.render(
            "Sensors: In-Cab Infrared Camera (PERCLOS/Yawn/Pose) | Bus CAN-Bus API (Drive Time)", True, (130, 145, 165))
        self.screen.blit(footer_text, (20, 360))

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        pygame.quit()
