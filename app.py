from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, g
import sqlite3
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import jsonify
 
# ML imports
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- App Config ---------------- #
app = Flask(__name__)
app.secret_key = "secret123"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- Database ---------------- #

DATABASE = "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def reset_db():
    """Drop and recreate users table with correct schema"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS users")
    c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            mobile TEXT,
            locality TEXT,
            status TEXT DEFAULT 'Inactive',
            role TEXT DEFAULT 'user',
            registered_date TEXT,
            registered_time TEXT,
            last_login_date TEXT,
            last_login_time TEXT
        )
    """)
    conn.commit()
    conn.close()


def init_db():
    """Ensure admins table exists with correct schema."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # ✅ Always recreate admins table with correct schema
    c.execute("DROP TABLE IF EXISTS admins")
    c.execute("""
        CREATE TABLE admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            username TEXT,
            email TEXT,
            password TEXT,
            role TEXT DEFAULT 'admin'
        )
    """)

    conn.commit()
    conn.close()


# ---------------- Load ML Model ---------------- #

MODEL_PATH = "model/flower_model.h5"
LABELS_PATH = "model/class_labels.npy"

model = None
class_labels = []
IMG_SIZE = (128, 128)  # match training size

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_labels = np.load(LABELS_PATH, allow_pickle=True)
    print("✅ Model and class labels loaded successfully.")
    print("Class mapping:", class_labels)

# ---------------- Routes ---------------- #

@app.route("/")
def home():
    return render_template("user_home.html")

# -------- Register -------- #

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form.get("email")
        password = request.form["password"]
        mobile = request.form.get("mobile")
        locality = request.form.get("locality")

        # Registration timestamp
        now = datetime.now()
        reg_date = now.strftime("%Y-%m-%d")
        reg_time = now.strftime("%H:%M:%S")

        try:
            with sqlite3.connect("users.db") as conn:
                c = conn.cursor()
                # ✅ Ensure columns exist
                c.execute("PRAGMA table_info(users)")
                columns = [info[1] for info in c.fetchall()]
                if "email" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN email TEXT")
                if "status" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'Inactive'")
                if "role" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
                if "registered_date" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN registered_date TEXT")
                if "registered_time" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN registered_time TEXT")
                if "last_login_date" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN last_login_date TEXT")
                if "last_login_time" not in columns:
                    c.execute("ALTER TABLE users ADD COLUMN last_login_time TEXT")

                # ✅ Insert user
                c.execute("""
                    INSERT INTO users (username, email, password, mobile, locality, status, role, registered_date, registered_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (username, email, password, mobile, locality, "Inactive", "user", reg_date, reg_time))
                conn.commit()

            flash("✅ You are registered successfully! Wait for admin activation.", "success")
            return redirect(url_for("user_dashboard"))

        except sqlite3.IntegrityError:
            flash("❌ Username or email already exists.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")

# --------------------------🔹 User Dashboard Route-----------------------#

@app.route("/user_dashboard")
def user_dashboard():
    return render_template("user_dashboard.html")

@app.route("/go_back_register")
def go_back_register():
    return redirect(url_for("home"))

# -------------------------User Login -------------------------------------------------------- #


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        # Hardcoded admin credentials
        if username == "admin" and password == "admin":
            session["admin"] = True
            session["username"] = "admin"
            flash("✅ Admin login successful!", "success")
            return redirect(url_for("admin_view"))

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password)).fetchone()

        if user:
            if user["status"] != "Active":
                flash("⚠️ Account not activated yet. Please wait for admin approval.", "warning")
                return redirect(url_for("login"))

            now = datetime.now()
            last_date = now.strftime("%Y-%m-%d")
            last_time = now.strftime("%H:%M:%S")
            db.execute("UPDATE users SET last_login_date=?, last_login_time=? WHERE id=?", (last_date, last_time, user["id"]))
            db.commit()

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["admin"] = False

            flash("✅ Login successful!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template("login.html")




#------------------------------------------------------- Admin manage view ---------------------------------------------#


@app.route("/admin_view")
def admin_view():
    if not session.get("admin"):
        flash("Access denied! Admins only.", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return render_template("admin_manage.html", users=users)




@app.route("/go_back_login")
def go_back_login():
    return redirect(url_for("home"))




# --------------------------- User Home ----------------------------------------------- #

@app.route("/user_home")
def user_home():
    if "username" not in session:
        flash("Please login first.", "danger")
        return redirect(url_for("login"))
    return render_template("user_home.html")

# ---------------------------------------- User List (Admin only) ----------------------------- #



@app.route("/user_list", methods=["GET", "POST"])
def user_list():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT id, username, email, status, role FROM users")
    rows = cur.fetchall()
    users = [{"id": r[0], "username": r[1], "email": r[2], "status": r[3], "role": r[4]} for r in rows]
    conn.close()
    return render_template("user_list.html", users=users)


@app.route("/activate/<int:user_id>")
def activate_user(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET status='Active' WHERE id=?", (user_id,))
        conn.commit()
    flash("Your account has been activated successfully!", "success")
    return redirect(url_for("user_list"))


@app.route("/deactivate/<int:user_id>")
def deactivate_user(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET status='Inactive' WHERE id=?", (user_id,))
        conn.commit()
    flash("Your account has been deactivated.", "warning")
    return redirect(url_for("user_list"))


@app.route("/delete/<int:user_id>")
def delete_user(user_id):
    try:
        with sqlite3.connect(DATABASE) as conn_user:
            cur_user = conn_user.cursor()
            cur_user.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn_user.commit()

        init_admin_db()
        with sqlite3.connect(ADMINS_DB) as conn_admin:
            cur_admin = conn_admin.cursor()
            cur_admin.execute("DELETE FROM admins WHERE user_id=?", (user_id,))
            conn_admin.commit()

        flash("Account deleted permanently from both databases.", "danger")
    except sqlite3.Error as e:
        flash(f"❌ Database error: {e}", "danger")
    return redirect(url_for("user_list"))


@app.route("/update_role/<int:user_id>", methods=["POST"])
def update_role(user_id):
    new_role = request.form.get("role")

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Fetch user
    user = c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()

    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404

    # Update role in users table
    c.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()

    # If role = Active → show actual password
    if new_role == "Active":
        visible_password = user["password"]
    else:
        visible_password = "*****"

    return {
        "status": "success",
        "role": new_role,
        "username": user["username"],
        "password": visible_password
    }

    # -------- ADMIN ROLE SELECTED --------
    if new_role == "admin":
        admin_exists = c.execute("SELECT * FROM admins WHERE user_id=?", (user_id,)).fetchone()

        if admin_exists:
            c.execute("""
                UPDATE admins SET username=?, email=?, password=?, role='admin'
                WHERE user_id=?
            """, (user["username"], user["email"], user["password"], user_id))
        else:
            c.execute("""
                INSERT INTO admins (user_id, username, email, password, role)
                VALUES (?, ?, ?, ?, 'admin')
            """, (user_id, user["username"], user["email"], user["password"]))

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "role": new_role,
            "message": "Details saved in admin database!"
        })

    # -------- USER ROLE SELECTED --------
    conn.close()
    return jsonify({
        "status": "success",
        "role": new_role,
        "message": "Details saved in user database!"
    })


# ------------------------ Logout -------------------------------------------- #


@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("admin", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("register"))

# -------------------------------- Admin Dashboard ---------------------------------- #


@app.route("/admin_dashboard")
def admin_dashboard():
    if "username" not in session:
        flash("Please login first.", "danger")
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")

# ------------------------------------ Upload & Predict ------------------------------- #


@app.route("/index", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part.", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file.", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        if model is None:
            flash("Model not found. Train the model first.", "danger")
            return redirect(url_for("upload"))

        # Prediction
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return render_template("result.html", filename=filename, prediction=predicted_class)

    return render_template("index.html")

# ----------------------------------------------- Train Model (Admin only) -------------------------------- #


@app.route("/train_model")
def train_model_route():
    global model, class_labels
    if "admin" not in session or not session.get("admin"):
        flash("Admin access required.", "danger")
        return redirect(url_for("admin_login"))
    try:
        train_dir = "dataset/train"
        img_size = (128, 128)
        batch_size = 32
        epochs = 15
        train_datagen = image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=img_size, batch_size=batch_size,
            class_mode="categorical", subset="training", shuffle=True
        )
        val_gen = train_datagen.flow_from_directory(
            train_dir, target_size=img_size, batch_size=batch_size,
            class_mode="categorical", subset="validation", shuffle=False
        )
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128,128,3), include_top=False, weights="imagenet"
        )
        input_layer = Input(shape=(128,128,3))
        x = base_model(input_layer, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        output = Dense(train_gen.num_classes, activation="softmax")(x)
        new_model = Model(inputs=input_layer, outputs=output)
        base_model.trainable = False
        new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                          loss="categorical_crossentropy", metrics=["accuracy"])
        callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
        new_model.fit(train_gen, validation_data=val_gen,
                      epochs=5, verbose=1, callbacks=callbacks)
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        new_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                          loss="categorical_crossentropy", metrics=["accuracy"])
        new_model.fit(train_gen, validation_data=val_gen,
                      epochs=epochs, verbose=1, callbacks=callbacks)
        os.makedirs("model", exist_ok=True)
        new_model.save("model/flower_model.h5")
        np.save("model/class_labels.npy", np.array(list(train_gen.class_indices.keys())))
        model = new_model
        class_labels = list(train_gen.class_indices.keys())
        flash("✅ Model trained and saved successfully!", "success")
    except Exception as e:
        flash(f"Training failed: {str(e)}", "danger")
    return redirect(url_for("admin_dashboard"))

# ---------------------------------------------Serve Uploaded Files ---------------------------------- #


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------------------------------------------------------------- Run ------------------------------------- #


if __name__ == "__main__":
    reset_db()
    init_db()
    app.run(debug=True)
