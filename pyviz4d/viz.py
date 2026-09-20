import vtk
from typing import List
from .earth import WGS84, earth_actor

class TemporalActor:
    """Base class for objects that evolve over time."""
    def __init__(self, actor: vtk.vtkActor):
        self.actor = actor

    def update(self, current_time: float):
        pass

import numpy as np

class Viewer4D:
    """
    4D Viewer mapping VTK to a time-series animation loop seamlessly.
    """
    def __init__(self, size=(1200, 900), bg_color=(0.15, 0.15, 0.15), nrows=1, ncols=1):
        self.nrows = nrows
        self.ncols = ncols

        # Renderer window setup
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetSize(*size)
        self.ren_win.SetDoubleBuffer(1)

        # Setup multiple renderers for the grid
        self.renderers = []
        for r in range(nrows):
            for c in range(ncols):
                ren = vtk.vtkRenderer()

                # Compute normalized viewport coordinates (xmin, ymin, xmax, ymax)
                xmin = c / ncols
                xmax = (c + 1) / ncols
                ymin = 1.0 - (r + 1) / nrows
                ymax = 1.0 - r / nrows

                ren.SetViewport(xmin, ymin, xmax, ymax)
                ren.SetBackground(*bg_color)

                # Share the camera with the first renderer so POV is linked across all views
                if len(self.renderers) > 0:
                    ren.SetActiveCamera(self.renderers[0].GetActiveCamera())

                self.ren_win.AddRenderer(ren)
                self.renderers.append(ren)

        # Backward compatibility / primary renderer
        self.ren = self.renderers[0]

        # Interactor setup
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)

        # Start with standard trackball camera style
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(self.style)

        # Store initial camera state for reset
        self.initial_camera_state = None

        # State
        self.temporal_actors: List[TemporalActor] = []
        self.current_time = 0.0
        self.is_playing = True
        self.time_speed = 1.0
        self._max_time = None
        self._loop = True
        self.time_slider = None

        # Recording state
        self._record_video_path = None
        self._record_frames_dir = None
        self._video_writer = None
        self._recorded_frame_count = 0
        self._record_max_frames = None

        self._setup_orientation_marker()

    def _setup_orientation_marker(self):
        self._axes = vtk.vtkAxesActor()
        self._om = vtk.vtkOrientationMarkerWidget()
        self._om.SetOrientationMarker(self._axes)
        self._om.SetInteractor(self.iren)
        self._om.SetViewport(0.0, 0.0, 0.2, 0.2)
        self._om.EnabledOn()
        self._om.InteractiveOff()

    def _save_initial_camera(self):
        """Save the initial camera position for resetting."""
        cam = self.ren.GetActiveCamera()
        self.initial_camera_state = {
            'position': cam.GetPosition(),
            'focal_point': cam.GetFocalPoint(),
            'view_up': cam.GetViewUp(),
            'view_angle': cam.GetViewAngle(),
            'clipping_range': cam.GetClippingRange()
        }

    def _reset_camera_to_initial(self):
        """Restore the camera to its initial state."""
        if self.initial_camera_state:
            cam = self.ren.GetActiveCamera()
            cam.SetPosition(*self.initial_camera_state['position'])
            cam.SetFocalPoint(*self.initial_camera_state['focal_point'])
            cam.SetViewUp(*self.initial_camera_state['view_up'])
            cam.SetViewAngle(self.initial_camera_state['view_angle'])
            cam.SetClippingRange(*self.initial_camera_state['clipping_range'])
            self.ren_win.Render()
        else:
            self.ren.ResetCamera()
            self.ren_win.Render()

    def _setup_global_hotkeys(self):
        """Configure general key bindings applicable to the Viewer."""

        def key_cb(obj, event):
            sym = obj.GetKeySym()
            if not sym:
                return
            key = sym.lower()

            # f: toggle fullscreen / maximize window
            if key == "f":
                try:
                    if not hasattr(self, "_is_maximized") or not self._is_maximized:
                        self._prev_size = self.ren_win.GetSize()
                        self._prev_pos = self.ren_win.GetPosition()
                        screen_size = self.ren_win.GetScreenSize()
                        self.ren_win.SetSize(screen_size[0], screen_size[1])
                        self.ren_win.SetPosition(0, 0)
                        self._is_maximized = True
                    else:
                        self.ren_win.SetSize(*self._prev_size)
                        self.ren_win.SetPosition(*self._prev_pos)
                        self._is_maximized = False
                    self.ren_win.Render()
                except Exception as e:
                    print(f"Fullscreen toggle failed: {e}")

            # r: reset camera to initial position
            elif key == "r":
                self._reset_camera_to_initial()

        self.iren.AddObserver("KeyPressEvent", key_cb, 1.0)

    def add_actor(self, actor, view=None):
        """
        Adds an actor to the scene.
        Args:
            actor: A TemporalActor or standard vtkActor.
            view: If None, adds to all grid views. If an integer, adds to that specific view index (row-major).
                  If a tuple (r, c), adds to that specific grid cell.
        """
        is_temporal = isinstance(actor, TemporalActor)
        vtk_actor = actor.actor if is_temporal else actor

        if is_temporal and actor not in self.temporal_actors:
            self.temporal_actors.append(actor)

        renderers_to_add = []
        if view is None:
            renderers_to_add = self.renderers
        elif isinstance(view, int):
            renderers_to_add = [self.renderers[view]]
        elif isinstance(view, tuple):
            r, c = view
            idx = r * self.ncols + c
            renderers_to_add = [self.renderers[idx]]

        for ren in renderers_to_add:
            ren.AddActor(vtk_actor)

    def enable_recording(self, video_path: str = None, frames_dir: str = None, fps: int = 60, max_frames: int = None):
        """
        Enables recording the animation to a video file and/or individual frames.

        Args:
            video_path: Path to the output video file (e.g., 'output.mp4').
            frames_dir: Directory to save individual frame images (e.g., 'frames/').
            fps: Frame rate for the output video.
            max_frames: Optional limit on the number of frames to record.
        """
        self._record_video_path = video_path
        self._record_frames_dir = frames_dir
        self._record_max_frames = max_frames
        self._recorded_frame_count = 0

        if self._record_video_path:
            import imageio
            # imageio automatically uses ffmpeg for mp4 and similar formats
            self._video_writer = imageio.get_writer(self._record_video_path, fps=fps)

        if self._record_frames_dir:
            import os
            os.makedirs(self._record_frames_dir, exist_ok=True)

        self._window_to_image_filter = vtk.vtkWindowToImageFilter()
        self._window_to_image_filter.SetInput(self.ren_win)
        self._window_to_image_filter.ReadFrontBufferOff() # Read back buffer to avoid capturing overlapping windows

    def _capture_frame(self):
        """Internal method to capture the current frame and write it to video/disk."""
        self._window_to_image_filter.Modified()
        self._window_to_image_filter.Update()

        image_data = self._window_to_image_filter.GetOutput()
        width, height, _ = image_data.GetDimensions()
        vtk_array = image_data.GetPointData().GetScalars()

        from vtk.util.numpy_support import vtk_to_numpy
        numpy_array = vtk_to_numpy(vtk_array)

        # VTK array is flat, reshape it to (height, width, channels)
        image = numpy_array.reshape(height, width, -1)
        # VTK's y-axis is bottom-up, flip it for standard image format
        image = np.flipud(image)

        if self._video_writer is not None:
            self._video_writer.append_data(image)

        if self._record_frames_dir is not None:
            import os
            import imageio
            frame_path = os.path.join(self._record_frames_dir, f"frame_{self._recorded_frame_count:05d}.png")
            imageio.imwrite(frame_path, image)

        self._recorded_frame_count += 1

    def add_playback_ui(self, max_time, loop=True):
        from .ui import create_slider, create_text_actor
        self._max_time = max_time
        self._loop = loop

        # Time Slider
        def time_cb(obj, event):
            if not self.is_playing:
                self.current_time = obj.GetRepresentation().GetValue()
                self._update_scene()

        self.time_slider = create_slider(self.iren, "Time", 0, max_time, 0, 0.1, time_cb)

        # Speed Slider
        def speed_cb(obj, event):
            self.time_speed = obj.GetRepresentation().GetValue()

        self.speed_slider = create_slider(self.iren, "Speed", 0.1, 5.0, 1.0, 0.25, speed_cb)
        self.time_speed = 1.0

        # Play/Pause Text Instruction
        self.play_text = create_text_actor("SPACE: Play | 'f': Fullscreen | 'r': Reset", 0.75, 0.4, font_size=16)
        self.ren.AddActor(self.play_text)

        def key_cb(obj, event):
            key = obj.GetKeySym()
            if key == "space":
                self.is_playing = not self.is_playing

        self.iren.AddObserver("KeyPressEvent", key_cb)

    def _update_scene(self):
        for actor in self.temporal_actors:
            actor.update(self.current_time)
        self.ren_win.Render()

    def _timer_callback(self, obj, event):
        if self.is_playing:
            self.current_time += self.time_speed

            if self._max_time is not None:
                if self.current_time > self._max_time:
                    if self._loop:
                        self.current_time = 0.0
                    else:
                        self.current_time = self._max_time
                        self.is_playing = False

            if self.time_slider is not None:
                self.time_slider.GetRepresentation().SetValue(self.current_time)

            self._update_scene()

            # Handle recording after rendering
            is_recording = bool(self._record_video_path) or bool(self._record_frames_dir)
            if is_recording:
                if self._record_max_frames is None or self._recorded_frame_count < self._record_max_frames:
                    self._capture_frame()
                elif self._video_writer is not None:
                    # Close the video writer when we hit the frame limit
                    self._video_writer.close()
                    self._video_writer = None
                    print(f"\n[PyViz4D] Finished recording {self._recorded_frame_count} frames to {self._record_video_path}")

    def start(self, timer_interval_ms=16): # 16ms ~= 60 FPS
        self.iren.Initialize()

        # Save the initial camera position immediately after initialization
        if self.initial_camera_state is None:
            self._save_initial_camera()

        # Setup global hotkeys
        self._setup_global_hotkeys()

        # Hook up the 4D timer
        self.iren.AddObserver('TimerEvent', self._timer_callback)
        self.iren.CreateRepeatingTimer(timer_interval_ms)

        self.ren_win.Render()
        self.iren.Start()

class EarthViewer4D(Viewer4D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_earth()

    def _setup_earth(self):
        self.earth = earth_actor()
        self.add_actor(self.earth)

        # Position camera perfectly for a WGS84 globe
        # Looking straight at lat=0, lon=0 from out in space
        camera = self.ren.GetActiveCamera()
        r = WGS84.a / 1e3
        camera.SetPosition(r * 4, 0, 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        self.ren.ResetCameraClippingRange()

        # Enforce zoom limits to prevent flying inside the Earth
        def enforce_zoom_limit(obj, event):
            cam = self.ren.GetActiveCamera()
            pos = np.array(cam.GetPosition())
            focal = np.array(cam.GetFocalPoint())

            # Distance from center of earth (origin)
            dist = np.linalg.norm(pos)

            # Minimum distance: Earth radius + ~20km cushion
            min_dist = r + 20.0

            if dist < min_dist:
                # Push the camera back out along its vector from the origin
                direction = pos / dist
                new_pos = direction * min_dist

                # If we push the camera, we should also push the focal point so panning doesn't break
                shift = new_pos - pos
                new_focal = focal + shift

                cam.SetPosition(new_pos[0], new_pos[1], new_pos[2])
                cam.SetFocalPoint(new_focal[0], new_focal[1], new_focal[2])
                self.ren.ResetCameraClippingRange()
                # Force an immediate render update so it doesn't flicker
                self.ren_win.Render()

        # Hook into ALL zoom-related events
        self.iren.AddObserver("InteractionEvent", enforce_zoom_limit)
        self.iren.AddObserver("MouseWheelForwardEvent", enforce_zoom_limit)
        self.iren.AddObserver("MouseWheelBackwardEvent", enforce_zoom_limit)
        # Add to the style as well to catch right-click zoom drags
        self.style.AddObserver("InteractionEvent", enforce_zoom_limit)
