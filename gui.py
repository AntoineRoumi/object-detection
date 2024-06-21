import glfw
import OpenGL.GL as gl
import imgui.core as im
from imgui.integrations.glfw import GlfwRenderer
import numpy as np

class ImguiTextWindow:
    def __init__(self, title: str, x: int, y: int, width: int, height: int) -> None:
        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.title: str = title
        self.is_shown: bool = True
        self.is_expand: bool = True
        self.text: str = ''

    def set_text(self, text: str) -> None:
        self.text = text

    def draw(self) -> None:
        if self.is_shown:
            im.set_next_window_position(self.x, self.y, condition=im.ONCE)
            im.set_next_window_size(self.width, self.height, condition=im.ONCE)
            self.is_expand, self.is_shown = im.begin(self.title, True)
            if self.is_expand:
                im.text(self.text)
            im.end()


class Texture:
    def __init__(self, width: int, height: int) -> None:
        self.texture: int = gl.glGenTextures(1)
        self.width: int = width
        self.height: int = height

    def update_image_from_mem(self, img: np.ndarray, img_w: int, img_h: int) -> None:
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_w, img_h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)

class FpsCounter:
    def __init__(self) -> None:
        self.fps: float = 0
        self.t0: float = glfw.get_time()
        self.t: float = 0.
        self.frames_count: int = 0

    def new_frame(self) -> None:
        self.t = glfw.get_time()
        if self.t - self.t0 > 1.0 or self.frames_count == 0:
            self.fps = self.frames_count / (self.t - self.t0)
            self.t0 = self.t
            self.frames_count = 0
        self.frames_count += 1

    def get_fps(self) -> float:
        return self.fps

class Window:
    def __init__(self, title: str, width: int, height: int) -> None:
        self.init(title, width, height)

    def make_context_current(self):
        glfw.make_context_current(self.window)

    def init(self, title: str, width: int, height: int) -> None:
        if not glfw.init():
            return None
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.width = width
        self.height = height
        self.window = glfw.create_window(self.width, self.height, title, None, None)
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
        return self.fps_counter.get_fps()

    def begin_drawing(self) -> None:
        self.fps_counter.new_frame()
        self.make_context_current()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        im.new_frame()

    def end_drawing(self) -> None:
        im.render()
        self.imgui_impl.render(im.get_draw_data())
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        self.imgui_impl.process_inputs()

    def draw_background_from_mem(self, img: np.ndarray, img_w: int, img_h: int):
        self.bg_tex.update_image_from_mem(img, img_w, img_h)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.bg_fbo)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.bg_tex.texture, 0)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glBlitFramebuffer(0, 0, self.bg_tex.width, self.bg_tex.height, 0, self.height, self.width, 0, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

    def draw_imgui_menu(self, items: list[tuple[str, list[str]]]) -> dict[str, bool]:
        values: dict[str, bool] = {}
        with im.begin_main_menu_bar() as main_menu_bar:
            if main_menu_bar.opened:
                for menu_content in items:
                    menu_name = menu_content[0].capitalize()
                    with im.begin_menu(menu_name, True) as menu:
                        if menu.opened:
                            for menu_item in menu_content[1]:
                                if len(menu_item) == 0:
                                    im.separator()
                                else:
                                    values[menu_item], _ = im.menu_item(menu_item.capitalize())
        return values

    def draw_imgui_text_window(self, imgui_window: ImguiTextWindow) -> None:
        imgui_window.draw()

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)
    
    def close(self) -> None:
        glfw.set_window_should_close(self.window, glfw.TRUE)

    def terminate(self) -> None:
        self.imgui_impl.shutdown()
        glfw.terminate()


