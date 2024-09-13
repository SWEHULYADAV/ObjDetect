import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import json
import hashlib
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variables
thres = 0.50
is_detection_running = False
saving_video = False

# Video capture setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Camera Error", "Error: Could not open camera.")
    exit()
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 70)

# Load class names from coco.names file
classNames = []
classFile = 'coco.names'
try:
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except Exception as e:
    messagebox.showerror("File Error", f"Error opening {classFile}: {e}")
    exit()

# Model setup
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
try:
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
except Exception as e:
    messagebox.showerror("Model Error", f"Error loading model: {e}")
    exit()

# GUI setup
root = tk.Tk()
root.title("Object Detection App")
root.geometry("1280x720")

# Function to create gradient background
def create_gradient(canvas, width, height, rgb_tuple=(255, 127, 0)):
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = int((rgb_tuple[0] * y) / height)
        g = int((rgb_tuple[1] * y) / height)
        b = int((rgb_tuple[2] * y) / height)
        gradient[y, :] = (r, g, b)
    return ImageTk.PhotoImage(image=Image.fromarray(gradient))

# Canvas for gradient background
canvas = tk.Canvas(root, width=1280, height=720)
canvas.pack(fill="both", expand=True)
gradient_image = create_gradient(canvas, 1280, 720)
canvas.create_image(0, 0, anchor="nw", image=gradient_image)

# Frames for layout
frame_left = tk.Frame(root, bg='white', padx=20, pady=20)
frame_left.place(relx=0.02, rely=0.5, anchor=tk.W)
frame_right = tk.Frame(root, bg='white', padx=20, pady=20)
frame_right.place(relx=0.98, rely=0.5, anchor=tk.E)

# Label for video stream
label = ttk.Label(frame_left)
label.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Listbox for detected objects
listbox = tk.Listbox(frame_left, height=10, width=50)
listbox.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

# OpenCV window setup
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 1280, 720)

# Plot setup
fig, ax = plt.subplots()
x_data = {className: [] for className in classNames}
y_data = {className: [] for className in classNames}
lines = {className: ax.plot([], [], label=className)[0] for className in classNames}
canvas_plot = FigureCanvasTkAgg(fig, master=frame_right)
canvas_plot.get_tk_widget().grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

# Function to start object detection
def start_detection():
    global is_detection_running
    is_detection_running = True
    update()

# Function to stop object detection
def stop_detection():
    global is_detection_running
    is_detection_running = False

# Function to toggle object detection on/off
def toggle_detection():
    global is_detection_running
    is_detection_running = not is_detection_running
    if is_detection_running:
        update()

# Function to start saving video
def start_saving_video():
    global saving_video, out
    saving_video = True
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Function to stop saving video
def stop_saving_video():
    global saving_video, out
    saving_video = False
    if out:
        out.release()

# Function to display FPS
def display_fps():
    global fps, frame_count, start_time
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        seconds = end_time - start_time
        fps = round(frame_count / seconds, 2)
        frame_count = 0
        start_time = time.time()
    if 'fps_label' in globals():
        fps_label.config(text=f"FPS: {fps}")
    root.after(100, display_fps)

# Function to change plot line colors
def change_plot_colors():
    global lines
    new_colors = simpledialog.askstring("Change Plot Colors", "Enter new colors (comma-separated RGB tuples, e.g., '255,0,0, 0,255,0'): ")
    if new_colors:
        try:
            color_list = [tuple(map(int, color.split(','))) for color in new_colors.split()]
            if len(color_list) == len(classNames):
                for i, className in enumerate(classNames):
                    lines[className].set_color(color_list[i])
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
                canvas_plot.draw()
            else:
                messagebox.showerror("Error", f"Number of colors ({len(color_list)}) should match the number of classes ({len(classNames)})")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid color format. Please use comma-separated RGB tuples.")
            print(e)

# Function to reset plot
def reset_plot():
    global x_data, y_data
    x_data = {className: [] for className in classNames}
    y_data = {className: [] for className in classNames}
    for className in classNames:
        lines[className].set_xdata([])
        lines[className].set_ydata([])
    ax.relim()
    ax.autoscale_view()
    canvas_plot.draw()

# Function to handle keyboard input
def on_key_press(event):
    if event.char == 'q':
        root.destroy()
        stop_detection()

# Bind keyboard events
root.bind('<KeyPress>', on_key_press)

# Function to update GUI with detected objects and plot
def update():
    global is_detection_running, saving_video, fps

    success, img = cap.read()
    if not success or img is None:
        print("Error: No frame captured.")
        return

    img = cv2.resize(img, (700, 500))

    if is_detection_running:
        try:
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            detected_objects = []
            current_time = time.time()

            for className in classNames:
                x_data[className].append(current_time)
                y_data[className].append(0)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if 0 <= classId < len(classNames):
                        className = classNames[classId - 1]
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        detected_objects.append(f"{className.upper()}: {round(confidence * 100, 2)}%")
                        y_data[className][-1] += 1

            listbox.delete(0, tk.END)
            for obj in detected_objects:
                listbox.insert(tk.END, obj)

            for className in classNames:
                lines[className].set_xdata(x_data[className])
                lines[className].set_ydata(y_data[className])
            ax.relim()
            ax.autoscale_view()
            canvas_plot.draw()

        except Exception as e:
            print(f"Error in object detection: {e}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    label.img = img_tk
    label.config(image=img_tk)

    if saving_video and out:
        try:
            out.write(cv2.cvtColor(np.array(Image.fromarray(img_rgb)), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error saving video: {e}")

    if is_detection_running:
        root.after(10, update)

    if 'fps_label' in globals():
        fps_label.config(text=f"FPS: {fps}")

# Start FPS monitoring
display_fps()

# User authentication and signup functions
def signup_user():
    try:
        username = simpledialog.askstring("Signup", "Enter username:")
        if not username:
            return
        password = simpledialog.askstring("Signup", "Enter password:")
        if not password:
            return

        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        signup_data = {
            "username": username,
            "password": hashed_password,
            "signup_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open('users.json', 'r') as f:
                users_data = json.load(f)
        except FileNotFoundError:
            users_data = {"users": []}

        for user in users_data["users"]:
            if user["username"] == username:
                messagebox.showerror("Signup Error", "Username already exists.")
                return

        users_data["users"].append(signup_data)

        with open('users.json', 'w') as f:
            json.dump(users_data, f, indent=4)
        messagebox.showinfo("Signup Success", "Signup successful. Please login to continue.")

    except Exception as e:
        messagebox.showerror("Signup Error", f"Error during signup: {e}")


def login_user():
    global is_detection_running
    if is_detection_running:
        messagebox.showerror("Login Error", "Please stop detection before logging in.")
        return

    username = simpledialog.askstring("Login", "Enter username:")
    if not username:
        return
    
    password = simpledialog.askstring("Login", "Enter password:")
    if not password:
        return

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        with open('users.json', 'r') as f:
            users_data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Login Error", "No users found. Please signup first.")
        return

    for user in users_data["users"]:
        if user["username"] == username and user["password"] == hashed_password:
            messagebox.showinfo("Login Success", f"Welcome, {username}!")
            # Start object detection after successful login
            start_detection()
            return

    messagebox.showerror("Login Error", "Invalid username or password.")

# GUI elements for user authentication
frame_auth = tk.Frame(root, bg='white', padx=20, pady=20)
frame_auth.place(relx=0.5, rely=0.02, anchor=tk.N)

auth_label = ttk.Label(frame_auth, text="User Authentication", font=("Helvetica", 16, "bold"))
auth_label.grid(row=0, column=0, columnspan=2, pady=10)

signup_button = ttk.Button(frame_auth, text="Signup", command=signup_user)
signup_button.grid(row=1, column=0, padx=10, pady=10)

login_button = ttk.Button(frame_auth, text="Login", command=login_user)
login_button.grid(row=1, column=1, padx=10, pady=10)

# Start Detection button
start_button = ttk.Button(frame_left, text="Start Detection", command=start_detection)
start_button.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

# Stop Detection button
stop_button = ttk.Button(frame_left, text="Stop Detection", command=stop_detection)
stop_button.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

# Toggle Detection button
toggle_button = ttk.Button(frame_left, text="Toggle Detection", command=toggle_detection)
toggle_button.grid(row=3, column=2, padx=10, pady=10, sticky="nsew")

# Start Saving Video button
save_button = ttk.Button(frame_left, text="Save Video", command=start_saving_video)
save_button.grid(row=3, column=3, padx=10, pady=10, sticky="nsew")

# Stop Saving Video button
stop_save_button = ttk.Button(frame_left, text="Stop Saving Video", command=stop_saving_video)
stop_save_button.grid(row=3, column=4, padx=10, pady=10, sticky="nsew")

# Display FPS label
fps_label = ttk.Label(frame_left, text=f"FPS: {fps}", font=("Helvetica", 12))
fps_label.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

# Change Plot Colors button
change_color_button = ttk.Button(frame_left, text="Change Plot Colors", command=change_plot_colors)
change_color_button.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

# Reset Plot button
reset_plot_button = ttk.Button(frame_left, text="Reset Plot", command=reset_plot)
reset_plot_button.grid(row=4, column=2, padx=10, pady=10, sticky="nsew")

# Main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
if out:
    out.release()
