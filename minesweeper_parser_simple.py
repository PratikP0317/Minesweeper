import tkinter as tk
from tkinter import ttk, messagebox
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import threading

class SimpleMinesweeperParser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Minesweeper Parser with OCR")
        self.root.geometry("800x600")
        
        # Variables
        self.captured_image = None
        self.selection_coords = None
        self.is_streaming = False
        self.current_board = None
        
        # Settings
        self.grid_rows = tk.IntVar(value=8)
        self.grid_cols = tk.IntVar(value=10)
        self.cell_width = tk.IntVar(value=170)
        self.cell_height = tk.IntVar(value=170)
        self.stream_fps = tk.IntVar(value=5)
        
        # Initialize OCR
        self.ocr_engine = None
        self.init_ocr()

        # Preview render meta (for click-to-calibrate mapping)
        self.preview_meta = {
            'scale': 1.0,
            'disp_w': 0,
            'disp_h': 0,
            'offset_x': 0,
            'offset_y': 0,
        }

        # Calibration state
        self.calibrating = False
        self.cal_class = tk.IntVar(value=1)  # default to number 1
        # samples: {label: [{'hsv':[h,s,v], 'lab':[L,a,b]}]}
        self.cal_samples = {}
        # prototypes: {label: {'hsv':[...], 'lab':[...]}}
        self.cal_prototypes = {}
        
        self.setup_ui()
        
    def init_ocr(self):
        """Initialize EasyOCR"""
        try:
            import easyocr
            print("Initializing EasyOCR...")
            self.ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR ready!")
        except Exception as e:
            print(f"❌ EasyOCR failed: {e}")
            print("📝 Install with: pip install easyocr")
            
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="1. Capture Area", 
                  command=self.capture_area).pack(side=tk.LEFT, padx=(0, 10))
        
        self.stream_btn = ttk.Button(controls_frame, text="2. Start Parsing", 
                                    command=self.toggle_stream)
        self.stream_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(controls_frame, text="3. Show in Game", 
                  command=self.show_in_game).pack(side=tk.LEFT, padx=(0, 10))
        
        # Grid settings
        grid_frame = ttk.LabelFrame(main_frame, text="Grid Settings", padding=10)
        grid_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(grid_frame, text="Rows:").grid(row=0, column=0)
        ttk.Spinbox(grid_frame, from_=1, to=200, textvariable=self.grid_rows, width=5,
                    command=self.update_preview).grid(row=0, column=1, padx=5)
        
        ttk.Label(grid_frame, text="Cols:").grid(row=0, column=2)
        ttk.Spinbox(grid_frame, from_=1, to=200, textvariable=self.grid_cols, width=5,
                    command=self.update_preview).grid(row=0, column=3, padx=5)
        
        ttk.Label(grid_frame, text="Cell W:").grid(row=0, column=4)
        ttk.Spinbox(grid_frame, from_=1, to=9999, textvariable=self.cell_width, width=6,
                    command=self.update_preview).grid(row=0, column=5, padx=5)
        
        ttk.Label(grid_frame, text="Cell H:").grid(row=0, column=6)
        ttk.Spinbox(grid_frame, from_=1, to=9999, textvariable=self.cell_height, width=6,
                    command=self.update_preview).grid(row=0, column=7, padx=5)
        
        # Status
        self.status = tk.StringVar(value="Ready - Click 'Capture Area' to start")
        ttk.Label(main_frame, textvariable=self.status, font=('Arial', 12)).pack(pady=10)
        
        # Preview
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(preview_frame, bg="white", width=600, height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Bind for calibration clicks
        self.canvas.bind('<Button-1>', self._on_canvas_click)

        # Calibration UI
        calib = ttk.LabelFrame(main_frame, text="Calibration (click inside preview to add samples)", padding=10)
        calib.pack(fill=tk.X, pady=(10, 0))

        classes = [(-2, 'Flag (-2)'), (-1, 'Unrevealed (-1)'), (0, 'Empty (0)'),
                   (1, 'One (1)'), (2, 'Two (2)'), (3, 'Three (3)'), (4, 'Four (4)')]
        r = 0
        for i, (val, text) in enumerate(classes):
            ttk.Radiobutton(calib, text=text, value=val, variable=self.cal_class).grid(row=r, column=i, sticky=tk.W, padx=6)
        r += 1
        ttk.Button(calib, text="Start Calibration", command=self._toggle_calibration).grid(row=r, column=0, padx=6)
        ttk.Button(calib, text="Clear", command=self._clear_calibration).grid(row=r, column=1, padx=6)
        ttk.Button(calib, text="Save", command=self._save_calibration).grid(row=r, column=2, padx=6)
        ttk.Button(calib, text="Load", command=self._load_calibration).grid(row=r, column=3, padx=6)
        self.cal_status = tk.StringVar(value="No samples")
        ttk.Label(calib, textvariable=self.cal_status).grid(row=r, column=4, padx=10, sticky=tk.W)

        # Also update preview when values are typed directly into spinboxes
        try:
            for v in (self.grid_rows, self.grid_cols, self.cell_width, self.cell_height):
                v.trace_add('write', lambda *args: self.update_preview())
        except Exception:
            pass
        
    def capture_area(self):
        """Simple area capture"""
        self.root.withdraw()
        time.sleep(1)  # Give user time to position
        
        messagebox.showinfo("Capture", "Click and drag to select the minesweeper board area.\nPress ESC to cancel.")
        
        try:
            # Create selection window
            selection_window = tk.Toplevel()
            selection_window.attributes('-fullscreen', True)
            selection_window.attributes('-alpha', 0.3)
            selection_window.configure(bg='black')
            selection_window.attributes('-topmost', True)
            
            canvas = tk.Canvas(selection_window, highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            start_x = start_y = end_x = end_y = 0
            selection_rect = None
            
            def on_click(event):
                nonlocal start_x, start_y, selection_rect
                start_x, start_y = event.x, event.y
                if selection_rect:
                    canvas.delete(selection_rect)
                    
            def on_drag(event):
                nonlocal end_x, end_y, selection_rect
                end_x, end_y = event.x, event.y
                if selection_rect:
                    canvas.delete(selection_rect)
                selection_rect = canvas.create_rectangle(start_x, start_y, end_x, end_y, 
                                                       outline='red', width=3)
                
            def on_release(event):
                nonlocal end_x, end_y
                end_x, end_y = event.x, event.y
                selection_window.destroy()
                self.process_selection(start_x, start_y, end_x, end_y)
                
            def on_escape(event):
                selection_window.destroy()
                self.root.deiconify()
                self.status.set("Capture cancelled")
                
            canvas.bind('<Button-1>', on_click)
            canvas.bind('<B1-Motion>', on_drag)
            canvas.bind('<ButtonRelease-1>', on_release)
            selection_window.bind('<Escape>', on_escape)
            
            # Instructions
            canvas.create_text(selection_window.winfo_screenwidth()//2, 50, 
                             text="Click and drag to select minesweeper area. Press ESC to cancel.", 
                             fill='white', font=('Arial', 16))
            
        except Exception as e:
            messagebox.showerror("Error", f"Capture failed: {e}")
            self.root.deiconify()
            
    def process_selection(self, x1, y1, x2, y2):
        """Process selected area"""
        try:
            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            if width < 50 or height < 50:
                messagebox.showwarning("Warning", "Area too small")
                self.root.deiconify()
                return
                
            self.selection_coords = (left, top, width, height)
            
            # Capture the area
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            self.captured_image = np.array(screenshot)
            
            self.status.set(f"✅ Captured {width}x{height} area. Click 'Start Parsing' to test OCR.")
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process selection: {e}")
        finally:
            self.root.deiconify()
            
    def update_preview(self):
        """Update preview with grid overlay"""
        if self.captured_image is None:
            return
            
        try:
            # Convert image
            img = Image.fromarray(cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB))
            
            # Draw grid
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            rows = self.grid_rows.get()
            cols = self.grid_cols.get()
            cell_w = self.cell_width.get()
            cell_h = self.cell_height.get()
            
            # Draw grid lines
            for i in range(cols + 1):
                x = i * cell_w
                if x <= img.width:
                    draw.line([(x, 0), (x, min(rows * cell_h, img.height))], fill='red', width=2)
            
            for i in range(rows + 1):
                y = i * cell_h
                if y <= img.height:
                    draw.line([(0, y), (min(cols * cell_w, img.width), y)], fill='red', width=2)
            
            # Scale to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            offset_x = 0
            offset_y = 0
            scale = 1.0
            disp_w, disp_h = img.width, img.height
            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / img.width, canvas_height / img.height, 1.0)
                new_size = (int(img.width * scale), int(img.height * scale))
                disp_w, disp_h = new_size
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                # Center within canvas
                offset_x = (canvas_width - new_size[0]) // 2
                offset_y = (canvas_height - new_size[1]) // 2
            
            # Display
            self.preview_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(offset_x, offset_y, image=self.preview_image, anchor=tk.NW)
            # Save meta for click mapping
            self.preview_meta.update({'scale': scale, 'disp_w': disp_w, 'disp_h': disp_h,
                                      'offset_x': offset_x, 'offset_y': offset_y})
            
        except Exception as e:
            print(f"Preview error: {e}")

    # ===== Calibration helpers =====
    def _toggle_calibration(self):
        self.calibrating = not self.calibrating
        self.status.set("Calibration: ON - click inside preview to sample" if self.calibrating else "Calibration: OFF")

    def _clear_calibration(self):
        self.cal_samples.clear()
        self.cal_prototypes.clear()
        self.cal_status.set("No samples")
        self.status.set("Calibration data cleared")

    def _save_calibration(self):
        try:
            import json
            data = self.cal_samples
            with open('ms_calibration.json', 'w') as f:
                json.dump(data, f, indent=2)
            self.status.set("Calibration saved to ms_calibration.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration: {e}")

    def _load_calibration(self):
        try:
            import json
            with open('ms_calibration.json', 'r') as f:
                self.cal_samples = json.load(f)
            # keys are strings in json; convert to int
            self.cal_samples = {int(k): v for k, v in self.cal_samples.items()}
            self._recompute_prototypes()
            self._update_cal_status()
            self.status.set("Calibration loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration: {e}")

    def _update_cal_status(self):
        total = sum(len(v) for v in self.cal_samples.values())
        classes = ', '.join(f"{k}:{len(v)}" for k, v in sorted(self.cal_samples.items()))
        self.cal_status.set(f"Samples {total} [{classes}]")

    def _recompute_prototypes(self):
        self.cal_prototypes = {}
        for label, samples in self.cal_samples.items():
            if not samples:
                continue
            hsvs = np.array([s['hsv'] for s in samples], dtype=np.float32)
            labs = np.array([s['lab'] for s in samples], dtype=np.float32)
            self.cal_prototypes[label] = {
                'hsv': np.mean(hsvs, axis=0).tolist(),
                'lab': np.mean(labs, axis=0).tolist(),
            }

    def _on_canvas_click(self, event):
        if not self.calibrating or self.captured_image is None:
            return
        # Map canvas click to image coordinate
        meta = self.preview_meta
        x = event.x - meta['offset_x']
        y = event.y - meta['offset_y']
        if x < 0 or y < 0 or x >= meta['disp_w'] or y >= meta['disp_h']:
            return  # outside image
        # Back to original image coordinates
        scale = meta['scale'] if meta['scale'] > 0 else 1.0
        img_x = int(x / scale)
        img_y = int(y / scale)
        h, w = self.captured_image.shape[:2]
        patch_r = 3
        x1 = max(0, img_x - patch_r)
        y1 = max(0, img_y - patch_r)
        x2 = min(w, img_x + patch_r + 1)
        y2 = min(h, img_y + patch_r + 1)
        patch = self.captured_image[y1:y2, x1:x2]
        if patch.size == 0:
            return
        # Compute mean HSV & Lab
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        hsv_mean = np.mean(hsv.reshape(-1, 3), axis=0).tolist()
        lab_mean = np.mean(lab.reshape(-1, 3), axis=0).tolist()
        label = int(self.cal_class.get())
        self.cal_samples.setdefault(label, []).append({'hsv': hsv_mean, 'lab': lab_mean})
        self._recompute_prototypes()
        self._update_cal_status()
        self.status.set(f"Sample added for class {label} at ({img_x},{img_y})")
    
    def _crop_inner_region(self, img, margin_px: int = 3):
        """Crop a small margin from all sides to ignore borders/shadows."""
        h, w = img.shape[:2]
        m = max(0, min(margin_px, min(h//6, w//6)))
        if m == 0:
            return img
        return img[m:h-m, m:w-m]

    def _classify_cell_by_color(self, cell_img):
        """Color-based classifier tailored for Google Minesweeper.
        Returns one of {1..8, 0, -1, -2} or None if unsure.
        -2 flag (red), 1 blue, 2 green, 3 red, 4 purple, 5 brown, 6 teal,
        7 black strokes, 8 gray strokes, 0 empty, -1 unrevealed.
        """
        try:
            # Work on inner region to avoid borders
            inner = self._crop_inner_region(cell_img, margin_px=3)
            if inner.size == 0:
                return None

            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            total_px = inner.shape[0] * inner.shape[1]
            if total_px == 0:
                return None

            # Helper to count percentage for a mask
            def pct(mask):
                return float(np.count_nonzero(mask)) / float(total_px)

            # Threshold helpers
            sat_hi = 60
            val_hi = 60

            # Red (flag) detection (two hue ranges in OpenCV: 0-10 and 170-179)
            red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
            red2 = cv2.inRange(hsv, (170, 70, 70), (179, 255, 255))
            red_pct = pct(cv2.bitwise_or(red1, red2))
            if red_pct > 0.06:  # >6% red pixels → flag
                return -2

            # Colored stroke masks (numbers)
            # Blue (1): H ~ [100,130]
            blue = cv2.inRange(hsv, (100, sat_hi, val_hi), (130, 255, 255))
            # Green (2): H ~ [45,85]
            green = cv2.inRange(hsv, (45, sat_hi, val_hi), (85, 255, 255))
            # Red (3): same as above but lower pixel share than flag => number
            red_num = cv2.bitwise_or(red1, red2)
            # Purple (4): H ~ [135,160]
            purple = cv2.inRange(hsv, (135, sat_hi, val_hi), (160, 255, 255))
            # Brown (5): H ~ [10,20] with decent saturation
            brown = cv2.inRange(hsv, (10, sat_hi, val_hi), (20, 255, 255))
            # Teal/Cyan (6): H ~ [85,100]
            teal = cv2.inRange(hsv, (85, sat_hi, val_hi), (100, 255, 255))

            # Black strokes (7): very low value
            black = cv2.inRange(hsv, (0, 0, 0), (179, 80, 60))
            # Gray strokes (8): low saturation mid value
            gray = cv2.inRange(hsv, (0, 0, 80), (179, 30, 200))

            color_scores = {
                1: pct(blue),
                2: pct(green),
                3: pct(red_num),
                4: pct(purple),
                5: pct(brown),
                6: pct(teal),
                7: pct(black),
                8: pct(gray),
            }

            # Decide number by highest colored score above a small threshold
            # Use higher threshold for gray/black to avoid false positives
            base_thresh = 0.015  # 1.5%
            gray_black_thresh = 0.03

            # Prefer colored digits 1-6 first
            colored_candidates = {k: v for k, v in color_scores.items() if k in {1, 2, 3, 4, 5, 6} and v > base_thresh}
            if colored_candidates:
                # If red dominates but not enough to be a flag, it is likely '3'
                return max(colored_candidates.items(), key=lambda kv: kv[1])[0]

            # Then try black 7 and gray 8
            if color_scores[7] > gray_black_thresh and color_scores[7] > color_scores[8] * 1.2:
                return 7
            if color_scores[8] > gray_black_thresh and color_scores[8] > color_scores[7] * 1.1:
                return 8

            # Distinguish unrevealed vs empty by brightness/variance
            inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            std_dev = float(np.std(inner_gray))
            mean_val = float(np.mean(inner_gray))
            if std_dev < 18 and 100 <= mean_val <= 200:
                return -1  # unrevealed
            if mean_val > 200:
                return 0  # empty

            return None
        except Exception:
            return None
            
    def recognize_cell_simple(self, cell_img):
        """Calibration-first, color-next, OCR-last recognition."""
        try:
            # 0) Use calibrated prototypes if available
            if self.cal_prototypes:
                res = self._classify_with_prototypes(cell_img)
                if res is not None:
                    return res

            # 1) Color-based quick classification
            color_guess = self._classify_cell_by_color(cell_img)
            if color_guess is not None:
                return color_guess

            # 2) OCR fallback for digits (1-8)
            # Convert to RGB for OCR
            if len(cell_img.shape) == 3:
                rgb_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cell_img
            
            # Resize for better OCR
            h, w = rgb_img.shape[:2]
            if h < 50 or w < 50:  # Make it bigger for OCR
                scale = max(3, 50 // min(h, w))
                rgb_img = cv2.resize(rgb_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
            # Use OCR to detect numbers
            if self.ocr_engine:
                try:
                    results = self.ocr_engine.readtext(rgb_img, detail=0, allowlist='12345678')
                    for result in results:
                        text = result.strip()
                        if text.isdigit() and 1 <= int(text) <= 8:
                            return int(text)
                except Exception as e:
                    print(f"OCR error: {e}")
            
            # 3) Last resort: simple brightness/variance heuristics
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) if len(rgb_img.shape) == 3 else rgb_img
            
            # Check if it's likely unrevealed (uniform gray)
            std_dev = np.std(gray)
            mean_val = np.mean(gray)
            
            if std_dev < 20 and 100 < mean_val < 200:  # Grayish and uniform
                return -1  # Unrevealed
            elif mean_val > 200:  # Very bright
                return 0   # Empty
            else:
                return -1  # Default to unrevealed
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return -1

    def _classify_with_prototypes(self, cell_img):
        """Classify using calibrated color prototypes.
        Returns label or None if uncertain.
        """
        try:
            # Use inner region and mask glyph-like pixels
            inner = self._crop_inner_region(cell_img, margin_px=4)
            if inner.size == 0:
                return None
            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            # Edge mask
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            edge_mask = (mag > 40).astype(np.uint8)
            # Colorfulness mask
            color_mask = ((hsv[:,:,1] > 60) & (hsv[:,:,2] > 60)).astype(np.uint8)
            mask = np.clip(edge_mask*2 + color_mask, 0, 1).astype(bool)
            if np.count_nonzero(mask) < inner.size // 50:  # if too small, relax to center area
                h2, w2 = inner.shape[:2]
                cy, cx = h2//2, w2//2
                ry, rx = max(2, h2//6), max(2, w2//6)
                mask = np.zeros((h2, w2), dtype=bool)
                mask[cy-ry:cy+ry+1, cx-rx:cx+rx+1] = True

            hsv_pixels = hsv.reshape(-1,3)[mask.reshape(-1)]
            lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB)
            lab_pixels = lab.reshape(-1,3)[mask.reshape(-1)]
            if hsv_pixels.size == 0 or lab_pixels.size == 0:
                return None
            hsv_mean = np.mean(hsv_pixels, axis=0)
            lab_mean = np.mean(lab_pixels, axis=0)

            def hsv_dist(a, b):
                # Hue wrap
                dh = min(abs(a[0]-b[0]), 180-abs(a[0]-b[0]))/180.0
                ds = abs(a[1]-b[1])/255.0
                dv = abs(a[2]-b[2])/255.0
                return dh*2.0 + ds*1.0 + dv*0.5

            def lab_dist(a, b):
                da = a-b
                return float(np.linalg.norm(da))/255.0

            best_label = None
            best_score = 1e9
            for label, proto in self.cal_prototypes.items():
                ph = np.array(proto['hsv'], dtype=np.float32)
                pl = np.array(proto['lab'], dtype=np.float32)
                d = hsv_dist(hsv_mean, ph) * 1.2 + lab_dist(lab_mean, pl) * 1.0
                if d < best_score:
                    best_score = d
                    best_label = label

            # Threshold to avoid wild guesses
            if best_score < 1.2:  # empirically tolerant
                return int(best_label)
            return None
        except Exception:
            return None
    
    def parse_board(self):
        """Parse the current board"""
        if self.captured_image is None:
            return None
            
        try:
            rows = self.grid_rows.get()
            cols = self.grid_cols.get()
            cell_w = self.cell_width.get()
            cell_h = self.cell_height.get()
            
            board = np.full((rows, cols), -1, dtype=int)
            
            print(f"\n📊 Parsing {rows}x{cols} board...")
            
            for row in range(rows):
                for col in range(cols):
                    # Extract cell
                    x1 = col * cell_w
                    y1 = row * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    
                    if x2 <= self.captured_image.shape[1] and y2 <= self.captured_image.shape[0]:
                        cell_img = self.captured_image[y1:y2, x1:x2]
                        cell_state = self.recognize_cell_simple(cell_img)
                        board[row, col] = cell_state
                        
                        # Print progress
                        if cell_state >= 0:
                            print(f"  Cell ({row},{col}): {cell_state}")
                        elif cell_state == -2:
                            print(f"  Cell ({row},{col}): FLAG")
                        
            return board
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def toggle_stream(self):
        """Toggle parsing"""
        if not self.is_streaming:
            if self.selection_coords is None:
                messagebox.showwarning("Warning", "Capture an area first!")
                return
                
            self.is_streaming = True
            self.stream_btn.config(text="Stop Parsing")
            self.status.set("🔄 Parsing...")
            
            # Parse once
            self.current_board = self.parse_board()
            
            if self.current_board is not None:
                # Count detected cells
                nums = np.sum((self.current_board >= 1) & (self.current_board <= 8))
                flags = np.sum(self.current_board == -2)
                empty = np.sum(self.current_board == 0)
                unrevealed = np.sum(self.current_board == -1)
                
                self.status.set(f"✅ Parsed! Numbers: {nums}, Flags: {flags}, Empty: {empty}, Unrevealed: {unrevealed}")
                print(f"\n📈 Results: Numbers={nums}, Flags={flags}, Empty={empty}, Unrevealed={unrevealed}")
                print("\n📋 Board:")
                print(self.current_board)
            else:
                self.status.set("❌ Parsing failed")
        else:
            self.is_streaming = False
            self.stream_btn.config(text="2. Start Parsing")
            self.status.set("⏹️ Stopped")
    
    def show_in_game(self):
        """Display parsed board matrix in a simple viewer"""
        if self.current_board is None:
            messagebox.showwarning("Warning", "Parse a board first!")
            return
            
        try:
            # Create display window
            display_window = tk.Toplevel(self.root)
            display_window.title("Parsed Board Matrix - Read Only")
            display_window.geometry("800x600")
            
            # Board info
            rows, cols = self.current_board.shape
            info_frame = ttk.Frame(display_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(info_frame, text=f"Board Size: {rows}x{cols}", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
            
            # Legend
            legend_frame = ttk.Frame(display_window)
            legend_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(legend_frame, text="Legend:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
            ttk.Label(legend_frame, text="1-8: Numbers | 0: Empty | -1: Unrevealed | -2: Flag", 
                     font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
            
            # Canvas for board display
            canvas_frame = ttk.Frame(display_window)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            canvas = tk.Canvas(canvas_frame, bg='white', relief=tk.SUNKEN, bd=2)
            v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
            h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
            
            canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Calculate cell size for display
            cell_size = max(30, min(50, 400 // max(rows, cols)))
            
            # Draw the board
            board_frame = tk.Frame(canvas, bg='white')
            
            # Color scheme
            colors = {
                -2: '#FF4444',  # Flag - Red
                -1: '#CCCCCC',  # Unrevealed - Gray
                0:  '#FFFFFF',  # Empty - White
                1:  '#0000FF',  # 1 - Blue
                2:  '#008000',  # 2 - Green
                3:  '#FF0000',  # 3 - Red
                4:  '#800080',  # 4 - Purple
                5:  '#800000',  # 5 - Maroon
                6:  '#008080',  # 6 - Teal
                7:  '#000000',  # 7 - Black
                8:  '#808080',  # 8 - Gray
            }
            
            text_colors = {
                -2: 'white',  # Flag
                -1: 'black',  # Unrevealed
                0:  'black',  # Empty
                1:  'white',  # Numbers
                2:  'white',
                3:  'white',
                4:  'white',
                5:  'white',
                6:  'white',
                7:  'white',
                8:  'white',
            }
            
            for row in range(rows):
                for col in range(cols):
                    value = self.current_board[row, col]
                    
                    # Cell background
                    cell_bg = colors.get(value, '#FFFFFF')
                    text_color = text_colors.get(value, 'black')
                    
                    # Create cell frame
                    cell_frame = tk.Frame(board_frame, bg=cell_bg, relief=tk.RAISED, bd=1,
                                        width=cell_size, height=cell_size)
                    cell_frame.grid(row=row, column=col, padx=1, pady=1)
                    cell_frame.grid_propagate(False)
                    
                    # Cell text
                    if value == -2:
                        text = "🚩"
                    elif value == -1:
                        text = "?"
                    elif value == 0:
                        text = ""
                    else:
                        text = str(value)
                    
                    label = tk.Label(cell_frame, text=text, bg=cell_bg, fg=text_color,
                                   font=('Arial', max(8, cell_size//4), 'bold'))
                    label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            
            # Add board to canvas
            canvas.create_window(0, 0, anchor=tk.NW, window=board_frame)
            
            # Update scroll region
            def update_scroll_region():
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
            
            display_window.after(100, update_scroll_region)
            
            # Matrix display
            matrix_frame = ttk.LabelFrame(display_window, text="Raw Matrix Data", padding=5)
            matrix_frame.pack(fill=tk.X, padx=10, pady=5)
            
            matrix_text = tk.Text(matrix_frame, height=8, font=('Courier', 10))
            matrix_scroll = ttk.Scrollbar(matrix_frame, orient=tk.VERTICAL, command=matrix_text.yview)
            matrix_text.configure(yscrollcommand=matrix_scroll.set)
            
            matrix_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            matrix_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Insert matrix data
            matrix_text.insert(tk.END, "Parsed Board Matrix:\n")
            matrix_text.insert(tk.END, str(self.current_board))
            matrix_text.insert(tk.END, f"\n\nShape: {self.current_board.shape}")
            matrix_text.insert(tk.END, f"\nUnique values: {np.unique(self.current_board)}")
            matrix_text.config(state=tk.DISABLED)
            
            # Buttons
            button_frame = ttk.Frame(display_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(button_frame, text="Save Matrix to File", 
                      command=lambda: self.save_matrix()).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="Close", 
                      command=display_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            self.status.set("📊 Board matrix displayed!")
            print(f"📊 Displaying {rows}x{cols} matrix in viewer")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display matrix: {e}")
            print(f"Display error: {e}")
    
    def save_matrix(self):
        """Save the parsed matrix to files"""
        if self.current_board is None:
            return
            
        try:
            # Save as numpy array
            np.save("parsed_board.npy", self.current_board)
            
            # Save as JSON for readability
            import json
            board_data = {
                'board': self.current_board.tolist(),
                'shape': self.current_board.shape,
                'unique_values': np.unique(self.current_board).tolist(),
                'timestamp': time.time()
            }
            
            with open("parsed_board.json", "w") as f:
                json.dump(board_data, f, indent=2)
            
            # Save as text
            with open("parsed_board.txt", "w") as f:
                f.write("Minesweeper Board Matrix\n")
                f.write("=" * 30 + "\n")
                f.write(f"Shape: {self.current_board.shape}\n")
                f.write(f"Legend: 1-8=Numbers, 0=Empty, -1=Unrevealed, -2=Flag\n\n")
                f.write(str(self.current_board))
            
            messagebox.showinfo("Success", "Matrix saved to:\n- parsed_board.npy\n- parsed_board.json\n- parsed_board.txt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save matrix: {e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("🚀 Starting Simple Minesweeper Parser...")
    
    # Check dependencies
    try:
        import easyocr
        print("✅ EasyOCR available")
    except ImportError:
        print("❌ EasyOCR not found. Install with: pip install easyocr")
        
    app = SimpleMinesweeperParser()
    app.run()
