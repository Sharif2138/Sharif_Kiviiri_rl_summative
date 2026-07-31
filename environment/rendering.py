import pygame
from OpenGL.GL import *
from OpenGL.GLU import *


class DriverFatigueRenderer:
    WIDTH, HEIGHT = 900, 560

    ACTION_LABELS = [
        ("0: MONITORING", (0.15, 0.65, 0.40)),
        ("1: MILD DRIVER VIBRATION", (0.95, 0.80, 0.10)),
        ("2: IN-CAB ALARM + RED LIGHTS", (0.90, 0.30, 0.25)),
        ("3: MANAGER CRITICAL ALARM", (0.60, 0.35, 0.75)),
    ]

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "AI Fleet Road Safety Monitor - 3D Cockpit View")
        pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                                pygame.DOUBLEBUF | pygame.OPENGL)

        self.clock = pygame.time.Clock()
        self.t = 0.0

        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_main = pygame.font.SysFont("Arial", 16)
        self.font_status = pygame.font.SysFont("Arial", 20, bold=True)

        self._init_gl()
        self.hud_texture_id = glGenTextures(1)

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.04, 0.05, 0.08, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.WIDTH / self.HEIGHT, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)

    def _draw_cuboid(self, sx, sy, sz):
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        vertices = [
            (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
            (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        ]
        faces = [
            (0, 1, 2, 3), (5, 4, 7, 6), (4, 0, 3, 7),
            (1, 5, 6, 2), (3, 2, 6, 7), (4, 5, 1, 0),
        ]
        glBegin(GL_QUADS)
        for face in faces:
            for idx in face:
                glVertex3f(*vertices[idx])
        glEnd()

    def _draw_road(self, drive_time):
        # Road surface
        glColor3f(0.12, 0.12, 0.14)
        glBegin(GL_QUADS)
        glVertex3f(-3.0, 0.0, 5.0)
        glVertex3f(3.0, 0.0, 5.0)
        glVertex3f(3.0, 0.0, -80.0)
        glVertex3f(-3.0, 0.0, -80.0)
        glEnd()

        # forward moving center line stripes
        offset = (self.t * 2.0 + drive_time * 0.1) % 4.0
        glColor3f(0.85, 0.75, 0.15)
        z = -80.0 + offset
        while z < 5.0:
            glBegin(GL_QUADS)
            glVertex3f(-0.08, 0.01, z)
            glVertex3f(0.08, 0.01, z)
            glVertex3f(0.08, 0.01, z + 1.2)
            glVertex3f(-0.08, 0.01, z + 1.2)
            glEnd()
            z += 4.0

        # Roadside perspective posts moving forward
        glColor3f(0.5, 0.15, 0.15)
        z_post = -80.0 + (offset * 2.5) % 10.0
        while z_post < 5.0:
            for x in (-3.4, 3.4):
                glPushMatrix()
                glTranslatef(x, 0.5, z_post)
                self._draw_cuboid(0.15, 1.0, 0.15)
                glPopMatrix()
            z_post += 10.0

    def _draw_gauge_bar(self, x, value, max_value, color):
        height = max(0.05, (value / max_value) * 1.6)
        glPushMatrix()
        glTranslatef(x, 0.05 + height / 2.0, 0.0)
        glColor3f(*color)
        self._draw_cuboid(0.35, height, 0.35)
        glPopMatrix()

    def _draw_beacon(self, color, spin):
        glPushMatrix()
        glTranslatef(0.0, 1.9, 0.0)
        glRotatef(spin, 0, 1, 0)
        glColor3f(*color)
        self._draw_cuboid(0.3, 0.3, 0.3)
        glPopMatrix()

    def _draw_dashboard(self, state, action):
        perclos, yawn, head_pose, _drive_time = state
        glPushMatrix()
        glTranslatef(0.0, 0.3, 1.6)

        # dashboard gauges: Red=PERCLOS, Yellow=Yawn, Blue=Head Pitch
        self._draw_gauge_bar(-1.2, perclos, 1.0, (0.85, 0.25, 0.25))
        self._draw_gauge_bar(-0.4, yawn, 10.0, (0.90, 0.70, 0.15))
        self._draw_gauge_bar(0.4, abs(head_pose), 90.0, (0.30, 0.55, 0.85))

        act_idx = action if action is not None and 0 <= action < len(
            self.ACTION_LABELS) else 0
        act_color = self.ACTION_LABELS[act_idx][1]
        spin_speed = 90.0 + act_idx * 60.0
        self._draw_beacon(act_color, spin=self.t * spin_speed)

        glPopMatrix()

    def _draw_hud(self, state, action, current_step, reward):
        perclos, yawn, head_pose, drive_time = state

        hud_surface = pygame.Surface(
            (self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 0))

        if perclos < 0.35:
            status_str, status_color = "STATUS: DRIVER ALERT", (46, 204, 113)
        elif perclos < 0.70:
            status_str, status_color = "STATUS: MILD FATIGUE DETECTED", (
                241, 196, 15)
        else:
            status_str, status_color = "STATUS: CRITICAL DROWSINESS WARNING!", (
                231, 76, 60)

        title_surf = self.font_title.render(
            "COMMERCIAL FLEET DRIVER MONITORING SYSTEM", True, (255, 255, 255))
        status_surf = self.font_status.render(status_str, True, status_color)

        hud_surface.blit(title_surf, (20, 15))
        hud_surface.blit(status_surf, (20, 55))

        telemetry_lines = [
            f"Shift Progress: Step {current_step}/50",
            f"Continuous Driving Time: {drive_time:.1f} Mins",
            f"PERCLOS (Eye Closure Rate): {perclos * 100:.1f}%",
            f"Yawn Frequency: {yawn:.1f} / min",
            f"Head Pitch Angle: {head_pose:.1f} deg",
            f"Step Reward: {reward:.2f}",
        ]

        for i, line in enumerate(telemetry_lines):
            surf = self.font_main.render(line, True, (210, 220, 235))
            hud_surface.blit(surf, (20, 100 + i * 24))

        act_idx = action if action is not None and 0 <= action < len(
            self.ACTION_LABELS) else 0
        act_label = self.ACTION_LABELS[act_idx][0]
        act_surf = self.font_status.render(
            f"AI ACTION -> {act_label}", True, (255, 255, 255))
        hud_surface.blit(act_surf, (20, self.HEIGHT - 40))

        texture_data = pygame.image.tostring(hud_surface, "RGBA", False)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.WIDTH, self.HEIGHT, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.hud_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.WIDTH,
                     self.HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1, 1, 1, 1)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(self.WIDTH, 0)
        glTexCoord2f(1, 1)
        glVertex2f(self.WIDTH, self.HEIGHT)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.HEIGHT)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def render(self, state, action, current_step, reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.t += 0.03
        _, _, _, drive_time = state

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(
            0.0, 1.6, 3.0,
            0.0, 0.6, -20.0,
            0.0, 1.0, 0.0
        )

        self._draw_road(drive_time)
        self._draw_dashboard(state, action)
        self._draw_hud(state, action, current_step, reward)

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        if glIsTexture(self.hud_texture_id):
            glDeleteTextures([self.hud_texture_id])
        pygame.quit()
