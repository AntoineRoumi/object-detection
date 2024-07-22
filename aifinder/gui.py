import glfw
import OpenGL.GL as gl
import imgui.core as im
from imgui.integrations.glfw import GlfwRenderer
import numpy as np


class ImguiTextWindow:
    """Wrapper class to facilitate creating Imgui text windows"""

    def __init__(self, title: str, x: int, y: int, width: int,
                 height: int) -> None:
        """title: title of the window
        x: initial x position of the window
        y: initial y position of the window
        width: initial width of the window
        height: initial height of the window"""

        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.title: str = title
        self.is_shown: bool = True
        self.is_expand: bool = True
        self.text: str = ''

    def set_text(self, text: str) -> None:
        """Set the text of the window.

        text: text to display in the window."""
        self.text = text

    def draw(self) -> None:
        """Draw the window in the curremt GLFW window."""
        if self.is_shown:
            im.set_next_window_position(self.x, self.y, condition=im.ONCE)
            im.set_next_window_size(self.width, self.height, condition=im.ONCE)
            self.is_expand, self.is_shown = im.begin(self.title)
            if self.is_expand:
                im.text(self.text)
            im.end()


class Texture:
    """Wrapper for OpenGL texture."""

    def __init__(self, width: int, height: int) -> None:
        """width: width of the texture
        height: height of the texture

        The size of the image is not necessarily the same as the image used for the texture."""

        self.texture: int = gl.glGenTextures(1)
        self.width: int = width
        self.height: int = height

    def update_image_from_mem(self, img: np.ndarray, img_w: int,
                              img_h: int, source_format = gl.GL_RGB) -> None:
        """Sets the image used for the texture from an image in memory.

        img: 3D Numpy array representing an RGB image
        img_w: width of the image
        img_h: height of the image"""

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, source_format, img_w, img_h, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)


class FpsCounter:
    """Helper class to calculate the FPS of a glfw window.
    The FPS value is calculated using the frame count of the last second."""

    def __init__(self) -> None:
        self.fps: float = 0
        self.t0: float = glfw.get_time()
        self.t: float = 0.
        self.frames_count: int = 0

    def new_frame(self) -> None:
        """Tells the FPS counter that a new frame has been displayed. 
        It then calculates the new FPS value."""

        self.t = glfw.get_time()
        if self.t - self.t0 > 1.0 or self.frames_count == 0:
            self.fps = self.frames_count / (self.t - self.t0)
            self.t0 = self.t
            self.frames_count = 0
        self.frames_count += 1

    def get_fps(self) -> float:
        """Returns the current FPS value."""

        return self.fps


class Window:
    """Class helping creating glfw and OpenGL window."""

    def __init__(self, title: str, width: int, height: int) -> None:
        """title: the title of the window
        width: width of the window
        height: height of the window"""

        self.init(title, width, height)

    def make_context_current(self) -> None:
        """Makes the Window the current OpenGL context."""

        glfw.make_context_current(self.window)

    def init(self, title: str, width: int, height: int) -> None:
        """Initializes the GLFW with the given parameters.

        title: title of the window 
        width: width of the GLFW window 
        height: height of the GLFW window"""

        if not glfw.init():
            return None
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.width = width
        self.height = height
        self.window = glfw.create_window(self.width, self.height, title, None,
                                         None)
        if not self.window:
            glfw.terminate()
            exit()
        self.make_context_current()

        im.create_context()
        self.imgui_impl = GlfwRenderer(self.window)

        self.bg_tex = Texture(width, height)
        self.bg_fbo = gl.glGenFramebuffers(1)

        self.fps_counter = FpsCounter()

    def get_fps(self) -> float:
        """Returns the FPS of the window, using a FpsCounter."""

        return self.fps_counter.get_fps()

    def begin_drawing(self) -> None:
        """Used to initialize the draw operations.
        MUST be called before performing any OpenGL/ImGui operation in the update loop."""

        self.fps_counter.new_frame()
        self.make_context_current()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        im.new_frame()

    def end_drawing(self) -> None:
        """Used to end the draw operations for the current frame.
        MUST be called after all the draw operations.
        Not calling it results in an error."""

        im.render()
        self.imgui_impl.render(im.get_draw_data())
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        self.imgui_impl.process_inputs()

    def draw_background_from_mem(self, img: np.ndarray, img_w: int,
                                 img_h: int, source_format = gl.GL_RGB):
        """Draw a texture on the background from an image stored in memory.

        img: RGB texture image represented as a 3D Numpy array
        img_w: width of the image 
        img_h: height of the image"""

        self.bg_tex.update_image_from_mem(img, img_w, img_h, source_format)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.bg_fbo)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER,
                                  gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D,
                                  self.bg_tex.texture, 0)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glBlitFramebuffer(0, 0, self.bg_tex.width, self.bg_tex.height, 0,
                             self.height, self.width, 0,
                             gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

    def draw_imgui_text_window(self, imgui_window: ImguiTextWindow) -> None:
        """Draw an ImguiTextWindow on the Window.

        imgui_window: the window to draw"""

        imgui_window.draw()

    def get_cursor_pos_in_window(self) -> tuple[float, float] | tuple[None, None]:
        self.cursor_x, self.cursor_y = glfw.get_cursor_pos(self.window)
        if 0. < self.cursor_x < self.width and 0. < self.cursor_y < self.height:
            return (self.cursor_x, self.cursor_y)
        else:
            return (None, None)

    def should_close(self) -> bool:
        """Returns whether closing the window has been requested."""

        return glfw.window_should_close(self.window)

    def close(self) -> None:
        """Closes the window."""

        glfw.set_window_should_close(self.window, glfw.TRUE)

    def terminate(self) -> None:
        """Terminates the underlying ImGui and GLFW libraries."""
        self.imgui_impl.shutdown()
        glfw.terminate()
