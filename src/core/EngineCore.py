"""EngineCore — application lifecycle, update/render loop."""
from __future__ import annotations

import sys
import math

import pygame

from src.core.Config import Config
from src.core.TimeSystem import TimeSystem
from src.core.RngSystem import RngSystem
from src.core.Logger import Logger, LogLevel

from src.math.Vec3 import Vec3
from src.math.Quat import Quat
from src.math.PlanetMath import PlanetMath

from src.render.Renderer import Renderer
from src.render.Camera import Camera
from src.render.MeshBuilder import MeshBuilder

from src.scene.Scene import Scene
from src.scene.Entity import Entity
from src.scene.Components import PlayerComponent, MeshComponent

from src.systems.AstroSystemStub import AstroSystemStub
from src.systems.ClimateSystemStub import ClimateSystemStub
from src.systems.GeoEventSystemStub import GeoEventSystemStub
from src.systems.PixelStageStub import PixelStageStub
from src.math.FloatingOrigin import FloatingOrigin

_TAG = "EngineCore"


class EngineCore:
    def __init__(self) -> None:
        self._config: Config | None = None
        self._time: TimeSystem | None = None
        self._rng: RngSystem | None = None
        self._renderer: Renderer | None = None
        self._camera: Camera | None = None
        self._scene: Scene | None = None
        self._floating_origin: FloatingOrigin | None = None

        # System stubs
        self._astro = AstroSystemStub()
        self._climate = ClimateSystemStub()
        self._geo = GeoEventSystemStub()
        self._pixel = PixelStageStub()

        # Player state
        self._player: Entity | None = None
        self._planet_radius: float = 1000.0
        self._walk_speed: float = 1.8
        self._frame: int = 0
        self._running: bool = False

    def init(self, config_path: str | None = None) -> None:
        Logger.set_level(LogLevel.INFO)
        self._config = Config(config_path)

        # Seed
        seed = int(self._config.get("seed", "default", default=42))
        self._rng = RngSystem(seed)
        Logger.info(_TAG, f"World seed: {seed}")

        # Time
        time_scale = float(self._config.get("day", "length_minutes", default=90)) / 90.0
        self._time = TimeSystem(game_time_scale=time_scale)

        # Planet params
        self._planet_radius = float(self._config.get("planet", "radius_units", default=1000))
        self._walk_speed = float(self._config.get("character", "walk_speed_units_per_s", default=1.8))

        # Renderer
        self._renderer = Renderer(1280, 720, "Dust — Stage 2: Planet Walk")
        self._renderer.init()

        # Camera
        self._camera = Camera(fov_deg=70.0, near=0.5, far=self._planet_radius * 10.0)
        surface_pos = Vec3(0.0, self._planet_radius + 1.8, 0.0)
        self._camera.local_position = surface_pos

        # Meshes
        sphere_mesh = MeshBuilder.uv_sphere(radius=self._planet_radius, stacks=24, slices=32)
        self._renderer.register_mesh("planet", sphere_mesh)
        indicator_mesh = MeshBuilder.uv_sphere(radius=1.5, stacks=8, slices=12)
        self._renderer.register_mesh("indicator", indicator_mesh)

        # Scene
        self._scene = Scene("PlanetWalkTest")

        planet_entity = Entity("Planet")
        planet_entity.add_component(MeshComponent("planet"))
        self._scene.add(planet_entity)

        self._player = Entity("Player")
        pc = PlayerComponent()
        pc.unit_dir = Vec3(0.0, 1.0, 0.0)  # Start at north pole / top
        self._player.add_component(pc)
        self._player.transform.position = surface_pos
        self._scene.add(self._player)

        # FloatingOrigin
        rebase_thresh = self._planet_radius * 0.5
        self._floating_origin = FloatingOrigin(
            rebase_threshold=rebase_thresh,
            planet_radius=self._planet_radius,
        )
        self._floating_origin.local_position = surface_pos

        Logger.info(_TAG, "EngineCore initialised")
        self._running = True

    def run(self) -> None:
        assert self._time and self._renderer and self._camera and self._scene
        self._time.reset()

        while self._running:
            self._time.tick()
            self._handle_input()
            self._update(self._time.game_dt)
            self._render()
            self._frame += 1

            if self._frame % 300 == 0:
                pc = self._player.get_component(PlayerComponent) if self._player else None
                fo = self._floating_origin
                if pc and fo:
                    ll = PlanetMath.from_direction(pc.unit_dir)
                    Logger.info(_TAG,
                        f"frame={self._frame} "
                        f"lat={math.degrees(ll.lat_rad):.2f}° "
                        f"lon={math.degrees(ll.lon_rad):.2f}° "
                        f"dt={self._time.real_dt*1000:.1f}ms"
                    )

    def _handle_input(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
            elif event.type == pygame.VIDEORESIZE:
                self._renderer.handle_resize(event.w, event.h)
                self._camera.set_aspect(event.w, event.h)

        keys = pygame.key.get_pressed()
        camera = self._camera
        pc = self._player.get_component(PlayerComponent) if self._player else None
        if not pc:
            return

        up = PlanetMath.up_at_position(camera.local_position)
        fwd_tangent = camera.forward_tangent()
        if fwd_tangent.is_near_zero():
            fwd_tangent = Vec3(1.0, 0.0, 0.0)
        right_tangent = fwd_tangent.cross(up).normalized()

        intent = Vec3.zero()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            intent = intent + fwd_tangent
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            intent = intent - fwd_tangent
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            intent = intent - right_tangent
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            intent = intent + right_tangent

        # Mouse look (horizontal only in this prototype)
        mouse_dx, _ = pygame.mouse.get_rel()
        if mouse_dx != 0:
            yaw_delta = mouse_dx * 0.002
            axis = up
            q = Quat.from_axis_angle(axis, -yaw_delta)
            camera.rotation = (q * camera.rotation).normalized()

        pc.intent = intent

    def _update(self, dt: float) -> None:
        assert self._player and self._camera and self._floating_origin
        pc = self._player.get_component(PlayerComponent)
        if not pc:
            return

        # Move player along sphere surface
        intent = pc.intent
        if not intent.is_near_zero():
            tangent = PlanetMath.tangent_forward(pc.unit_dir, intent)
            arc_per_frame = self._walk_speed / self._planet_radius * dt
            pc.unit_dir = PlanetMath.move_along_surface(pc.unit_dir, tangent, arc_per_frame)

        # Keep camera on surface
        surface_pos = pc.unit_dir * (self._planet_radius + 1.8)
        self._camera.local_position = surface_pos
        self._floating_origin.set_local_position(surface_pos)
        self._floating_origin.update_geo()

        # Align camera up to planet up
        self._camera.align_to_surface(dt)

        # Update system stubs
        gt = self._time.game_time_accum if self._time else 0.0
        self._astro.update(gt)
        self._climate.update(gt)
        self._geo.update(gt)

        # FloatingOrigin rebase
        self._floating_origin.try_rebase([])

    def _render(self) -> None:
        assert self._renderer and self._camera and self._player
        self._renderer.begin_frame()
        self._renderer.set_camera(self._camera)

        # Draw planet sphere
        self._renderer.draw_mesh("planet", color=(0.55, 0.45, 0.35))

        # Draw debug indicators at player position
        pc = self._player.get_component(PlayerComponent)
        if pc:
            pos = pc.unit_dir * (self._planet_radius + 1.8)
            up_vec = PlanetMath.up_at_position(pos)
            fwd_vec = self._camera.forward_tangent()

            # Up arrow (green)
            self._renderer.draw_line(pos, pos + up_vec * 10.0, color=(0.0, 1.0, 0.0))
            # Forward tangent (blue)
            if not fwd_vec.is_near_zero():
                self._renderer.draw_line(pos, pos + fwd_vec * 10.0, color=(0.0, 0.5, 1.0))
            # Player dot
            self._renderer.draw_point(pos, color=(1.0, 1.0, 0.0), size=8.0)

        self._renderer.end_frame()

    def shutdown(self) -> None:
        if self._renderer:
            self._renderer.shutdown()
        Logger.info(_TAG, "Shutdown complete")
