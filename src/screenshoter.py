import tkinter as tk
from PIL import ImageGrab, Image
import os

save_folder = '../dataset/datasets/screenshots'
object_name = 'pallet'

class ScreenshotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Square Screenshot Tool")

        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.canvas = tk.Canvas(root, cursor="cross", bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_drag(self, event):
        if self.rect:
            dx = event.x - self.start_x
            dy = event.y - self.start_y

            size = max(abs(dx), abs(dy))

            x2 = self.start_x + size if dx > 0 else self.start_x - size
            y2 = self.start_y + size if dy > 0 else self.start_y - size

            self.canvas.coords(
                self.rect,
                self.start_x, self.start_y,
                x2, y2
            )

    def on_release(self, event):
        if not self.rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.rect)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        self.root.withdraw()
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        self.root.deiconify()

        screenshot = screenshot.resize((640, 640), Image.LANCZOS)
        file_path = os.path.join(self.save_folder, f"{object_name}_{len(os.listdir(self.save_folder)) + 1}.png")
        screenshot.save(file_path)

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenshotApp(root)
    root.mainloop()