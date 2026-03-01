"""Renderer — pygame window + OpenGL render loop."""
from __future__ import annotations

import math
import ctypes

import pygame
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glDisable,
    glBegin, glEnd, glVertex3f, glNormal3f, glColor3f,
    glMatrixMode, glLoadMatrixf, glLoadIdentity, glViewport,
    glLineWidth, glPointSize,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_TRIANGLES, GL_LINES, GL_POINTS,
    GL_PROJECTION, GL_MODELVIEW,
    glLightfv, glEnable as glEnableF, GL_LIGHTING, GL_LIGHT0,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
    GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    glColorMaterial, GL_SMOOTH, glShadeModel,
)
from OpenGL.arrays import vbo

from src.math.Vec3 import Vec3
from src.math.Mat4 import Mat4
from src.render.Camera import Camera
from src.render.MeshBuilder import Mesh
from src.core.Logger import Logger

_TAG = "Renderer"


class Renderer:
    def __init__(self, width: int = 1280, height: int = 720, title: str = "Dust — Stage 2") -> None:
        self.width = width
        self.height = height
        self.title = title
        self._surface: pygame.Surface | None = None
        self._meshes: dict[str, Mesh] = {}
        self._running = False

    def init(self) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        self._surface = pygame.display.set_mode((self.width, self.height), flags)
        self._setup_gl()
        Logger.info(_TAG, f"Window {self.width}x{self.height} initialised")
        self._running = True

    def _setup_gl(self) -> None:
        glClearColor(0.02, 0.02, 0.05, 1.0)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.7, 1.0, 0.5, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.85, 0.6, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.1, 0.1, 0.15, 1.0])
        glViewport(0, 0, self.width, self.height)

    def register_mesh(self, name: str, mesh: Mesh) -> None:
        self._meshes[name] = mesh

    def begin_frame(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_camera(self, camera: Camera) -> None:
        camera.set_aspect(self.width, self.height)
        proj = camera.projection_matrix().to_list()
        view = camera.view_matrix().to_list()
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(proj)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(view)

    def draw_mesh(self, mesh_id: str, color: tuple = (0.72, 0.58, 0.45)) -> None:
        mesh = self._meshes.get(mesh_id)
        if mesh is None:
            return
        glColor3f(*color)
        glBegin(GL_TRIANGLES)
        for i in range(0, len(mesh.indices), 3):
            for k in range(3):
                idx = mesh.indices[i + k]
                n = mesh.normals[idx]
                v = mesh.vertices[idx]
                glNormal3f(*n)
                glVertex3f(*v)
        glEnd()

    def draw_line(self, a: Vec3, b: Vec3, color: tuple = (1.0, 1.0, 1.0)) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(a.x, a.y, a.z)
        glVertex3f(b.x, b.y, b.z)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_point(self, p: Vec3, color: tuple = (1.0, 1.0, 0.0), size: float = 6.0) -> None:
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glPointSize(size)
        glBegin(GL_POINTS)
        glVertex3f(p.x, p.y, p.z)
        glEnd()
        glEnable(GL_LIGHTING)

    def end_frame(self) -> None:
        pygame.display.flip()

    def handle_resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def shutdown(self) -> None:
        self._running = False
        pygame.quit()
        Logger.info(_TAG, "Renderer shutdown")
