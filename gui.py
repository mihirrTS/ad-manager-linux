import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
from PIL import Image, ImageTk
import os
import shutil
import subprocess
import threading
import cv2
import sys
import unicodedata

ADS_FOLDER = "ads"
CATEGORIES = [
    "male_kid", "male_teen", "male_adult",
    "female_kid", "female_teen", "female_adult"
]

class AdManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Targeted Ad Manager")
        self.root.geometry("1000x750")
        self.presentation_process = None
        self.category_var = tk.StringVar(value=CATEGORIES[0])
        self.theme = "light"
        self.setup_styles()
        self.build_ui()

    def setup_styles(self):
        self.bg_color = "#ffffff" if self.theme == "light" else "#1e1e1e"
        self.fg_color = "#000000" if self.theme == "light" else "#ffffff"
        self.entry_bg = "#f0f0f0" if self.theme == "light" else "#2e2e2e"
        self.status_bg = "#f0f0f0" if self.theme == "light" else "#2b2b2b"
        self.root.configure(bg=self.bg_color)

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.setup_styles()
        for widget in self.main_frame.winfo_children():
            try:
                widget.configure(bg=self.bg_color, fg=self.fg_color)
            except:
                pass
        self.status_text.configure(bg=self.status_bg, fg=self.fg_color)

    def build_ui(self):
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Button(self.main_frame, text="Toggle Light/Dark Mode", command=self.toggle_theme,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=5)

        tk.Label(self.main_frame, text="Select Category:", bg=self.bg_color, fg=self.fg_color).pack()
        self.dropdown = tk.OptionMenu(self.main_frame, self.category_var, *CATEGORIES)
        self.dropdown.pack()

        tk.Label(self.main_frame, text="Video Speed (0.5x to 2.0x):", bg=self.bg_color, fg=self.fg_color).pack()
        self.speed_slider = tk.Scale(self.main_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                     bg=self.bg_color, fg=self.fg_color)
        self.speed_slider.set(1.0)
        self.speed_slider.pack()

        tk.Label(self.main_frame, text="Video Play Duration (seconds):", bg=self.bg_color, fg=self.fg_color).pack()
        self.duration_slider = tk.Scale(self.main_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                        bg=self.bg_color, fg=self.fg_color)
        self.duration_slider.set(10)
        self.duration_slider.pack()

        tk.Button(self.main_frame, text="Upload Ad", command=self.upload_ad,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=5)

        tk.Button(self.main_frame, text="View Ads in Category", command=self.view_ads,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=2)

        tk.Button(self.main_frame, text="Start Presentation", command=self.run_presentation,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=2)

        tk.Button(self.main_frame, text="Stop Presentation", command=self.stop_presentation,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=2)

        self.ads_listbox = Listbox(self.main_frame, width=80, height=12)
        self.ads_listbox.pack(pady=10)
        self.ads_listbox.bind("<<ListboxSelect>>", self.display_thumbnail)

        tk.Button(self.main_frame, text="Delete Selected Ad", command=self.delete_ad,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=2)

        tk.Button(self.main_frame, text="Move Selected Ad to...", command=self.move_ad,
                  bg=self.bg_color, fg=self.fg_color).pack(pady=2)

        self.thumbnail_label = tk.Label(self.main_frame, bg=self.bg_color)
        self.thumbnail_label.pack()

        tk.Label(self.main_frame, text="Status & Detection Info:", bg=self.bg_color, fg=self.fg_color).pack()
        self.status_text = tk.Text(self.main_frame, height=10, bg=self.status_bg, fg=self.fg_color, state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def display_status(self, message):
        clean = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore').decode('utf-8')
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, clean + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def upload_ad(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not filepath:
            return
        category = self.category_var.get()
        dest_folder = os.path.join(ADS_FOLDER, category)
        os.makedirs(dest_folder, exist_ok=True)
        try:
            shutil.copy(filepath, dest_folder)
            messagebox.showinfo("Success", f"Ad uploaded to {category} category!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def view_ads(self):
        self.ads_listbox.delete(0, tk.END)
        category = self.category_var.get()
        folder_path = os.path.join(ADS_FOLDER, category)
        if not os.path.exists(folder_path):
            return
        ads = os.listdir(folder_path)
        for ad in ads:
            self.ads_listbox.insert(tk.END, ad)

    def display_thumbnail(self, event):
        selected = self.ads_listbox.curselection()
        if not selected:
            return
        ad_name = self.ads_listbox.get(selected[0])
        category = self.category_var.get()
        video_path = os.path.join(ADS_FOLDER, category, ad_name)
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((320, 240))
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_label.configure(image=photo)
                self.thumbnail_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def delete_ad(self):
        selected = self.ads_listbox.curselection()
        if not selected:
            return
        ad_name = self.ads_listbox.get(selected[0])
        category = self.category_var.get()
        path = os.path.join(ADS_FOLDER, category, ad_name)
        if messagebox.askyesno("Confirm Delete", f"Delete '{ad_name}'?"):
            try:
                os.remove(path)
                self.view_ads()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def move_ad(self):
        selected = self.ads_listbox.curselection()
        if not selected:
            return
        ad_name = self.ads_listbox.get(selected[0])
        current_category = self.category_var.get()
        src = os.path.join(ADS_FOLDER, current_category, ad_name)
        new_folder = filedialog.askdirectory(initialdir=ADS_FOLDER)
        if new_folder and messagebox.askyesno("Move", f"Move '{ad_name}' to this folder?"):
            try:
                shutil.move(src, os.path.join(new_folder, ad_name))
                self.view_ads()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run_presentation(self):
        speed = self.speed_slider.get()
        duration = self.duration_slider.get()
        self.display_status("📺 Starting presentation...")

        def start():
            try:
                self.presentation_process = subprocess.Popen(
                    [sys.executable, "main.py", str(speed), str(duration)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                for line in self.presentation_process.stdout:
                    if line.strip():
                        self.display_status(line.strip())
            except Exception as e:
                self.display_status(f"[ERROR] {e}")

        threading.Thread(target=start, daemon=True).start()

    def stop_presentation(self):
        if self.presentation_process:
            self.presentation_process.terminate()
            self.presentation_process = None
            messagebox.showinfo("Stopped", "Presentation stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdManagerApp(root)
    root.mainloop()
