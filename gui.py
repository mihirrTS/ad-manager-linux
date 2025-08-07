import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, scrolledtext
from PIL import Image, ImageTk
import os
import shutil
import subprocess
import threading
import cv2
import mediapipe as mp
import sys

ADS_FOLDER = "ads"
os.makedirs(ADS_FOLDER, exist_ok=True)
CATEGORIES = [
    "male_kid", "male_teen", "male_adult",
    "female_kid", "female_teen", "female_adult"
]


class AdManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Targeted Ad Manager")
        self.root.geometry("900x750")

        self.presentation_process = None
        self.dark_mode = tk.BooleanVar(value=False)
        self.category_var = tk.StringVar(value=CATEGORIES[0])
        self.show_terminal = tk.BooleanVar(value=True)

        self.build_ui()
        self.apply_theme()

    def build_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.main_frame, text="Select Category:").pack(pady=5)
        self.dropdown = tk.OptionMenu(self.main_frame, self.category_var, *CATEGORIES)
        self.dropdown.pack()

        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(pady=5)

        tk.Checkbutton(control_frame, text="Dark Mode", variable=self.dark_mode, command=self.toggle_theme).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(control_frame, text="Show Terminal", variable=self.show_terminal, command=self.toggle_terminal).pack(side=tk.LEFT, padx=5)

        # NEW: Speed and Duration Sliders
        tk.Label(self.main_frame, text="Video Speed (0.5x to 2.0x):").pack(pady=2)
        self.speed_slider = tk.Scale(self.main_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.speed_slider.set(1.0)
        self.speed_slider.pack()

        tk.Label(self.main_frame, text="Video Play Duration (seconds):").pack(pady=2)
        self.duration_slider = tk.Scale(self.main_frame, from_=1, to=15, orient=tk.HORIZONTAL)
        self.duration_slider.set(5)
        self.duration_slider.pack()

        tk.Button(self.main_frame, text="Upload Ad", command=self.upload_ad).pack(pady=5)
        tk.Button(self.main_frame, text="View Ads in Category", command=self.view_ads).pack(pady=5)
        tk.Button(self.main_frame, text="Start Presentation", command=self.run_presentation).pack(pady=5)
        tk.Button(self.main_frame, text="Stop Presentation", command=self.stop_presentation).pack(pady=5)

        self.ads_listbox = Listbox(self.main_frame, width=60)
        self.ads_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        self.ads_listbox.bind("<<ListboxSelect>>", self.display_thumbnail)

        tk.Button(self.main_frame, text="Delete Selected Ad", command=self.delete_ad).pack(pady=5)
        tk.Button(self.main_frame, text="Move Selected Ad to...", command=self.move_ad).pack(pady=5)

        self.thumbnail_label = tk.Label(self.main_frame)
        self.thumbnail_label.pack(pady=10)

        self.log_label = tk.Label(self.main_frame, text="Detection Logs:")
        self.log_label.pack(pady=5)
        self.log_box = scrolledtext.ScrolledText(self.main_frame, height=10, width=100, state='disabled')
        self.log_box.pack(pady=5, fill=tk.BOTH, expand=True)

    def toggle_theme(self):
        self.apply_theme()

    def toggle_terminal(self):
        if self.show_terminal.get():
            self.log_box.pack(pady=5, fill=tk.BOTH, expand=True)
            self.log_label.pack(pady=5)
        else:
            self.log_box.pack_forget()
            self.log_label.pack_forget()

    def apply_theme(self):
        dark = self.dark_mode.get()
        bg = "#2e2e2e" if dark else "#f0f4f8"
        fg = "white" if dark else "#000000"
        self.root.configure(bg=bg)
        self._apply_recursive_theme(self.root, bg, fg)

    def _apply_recursive_theme(self, widget, bg, fg):
        try:
            widget.configure(bg=bg, fg=fg)
        except:
            try:
                widget.configure(bg=bg)
            except:
                pass
        for child in widget.winfo_children():
            self._apply_recursive_theme(child, bg, fg)

    def log(self, message):
        print(message)
        if self.show_terminal.get():
            self.log_box.config(state='normal')
            self.log_box.insert(tk.END, message + "\n")
            self.log_box.yview(tk.END)
            self.log_box.config(state='disabled')

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
            messagebox.showinfo("Info", "No ads in this category.")
            return

        ads = os.listdir(folder_path)
        if not ads:
            messagebox.showinfo("Info", "No ads in this category.")
        else:
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
            messagebox.showerror("Error", f"Unable to load thumbnail: {e}")

    def delete_ad(self):
        selected = self.ads_listbox.curselection()
        if not selected:
            return
        ad_name = self.ads_listbox.get(selected[0])
        category = self.category_var.get()
        video_path = os.path.join(ADS_FOLDER, category, ad_name)

        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{ad_name}'?")
        if not confirm:
            return

        try:
            os.remove(video_path)
            messagebox.showinfo("Deleted", f"{ad_name} deleted.")
            self.view_ads()
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete: {e}")

    def move_ad(self):
        selected = self.ads_listbox.curselection()
        if not selected:
            return
        ad_name = self.ads_listbox.get(selected[0])
        current_category = self.category_var.get()
        src_path = os.path.join(ADS_FOLDER, current_category, ad_name)

        new_category = filedialog.askdirectory(initialdir=ADS_FOLDER, title="Select Destination Category Folder")
        if new_category:
            confirm = messagebox.askyesno("Confirm Move", f"Move '{ad_name}' to selected folder?")
            if not confirm:
                return
            try:
                shutil.move(src_path, os.path.join(new_category, ad_name))
                messagebox.showinfo("Moved", f"{ad_name} moved.")
                self.view_ads()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run_presentation(self):
        self.log("🔄 Presentation is starting...")

        speed = self.speed_slider.get()
        duration = self.duration_slider.get()

        def start():
            try:
                self.presentation_process = subprocess.Popen(
                    [sys.executable, "main.py", str(speed), str(duration)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                for line in self.presentation_process.stdout:
                    self.log(line.strip())
            except Exception as e:
                self.log(f"[Error] {e}")

        threading.Thread(target=start, daemon=True).start()

    def stop_presentation(self):
        if self.presentation_process:
            self.presentation_process.terminate()
            self.presentation_process = None
            messagebox.showinfo("Stopped", "Presentation stopped.")
        else:
            messagebox.showinfo("Info", "Presentation is not running.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AdManagerApp(root)
    root.mainloop()
