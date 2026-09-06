import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


def draw_realistic_worker(draw: ImageDraw.Draw, x: int, y: int, scale: float = 1.0, has_hardhat: bool = True, has_vest: bool = True):
    """
    Renders a realistic worker sprite with hardhat, safety vest, and overalls.
    (x, y) is the bottom-center anchor of the worker.
    """
    w = int(45 * scale)
    h = int(140 * scale)
    top_y = y - h
    left_x = x - w // 2
    right_x = x + w // 2

    # 1. Legs / Trousers (Dark Blue)
    leg_w = w // 2 - 2
    draw.rectangle([(left_x, y - int(h * 0.45)), (left_x + leg_w, y)], fill=(30, 45, 75))
    draw.rectangle([(right_x - leg_w, y - int(h * 0.45)), (right_x, y)], fill=(30, 45, 75))
    # Boots (Black/Brown)
    draw.rectangle([(left_x - 3, y - 12), (left_x + leg_w + 3, y)], fill=(20, 20, 20))
    draw.rectangle([(right_x - leg_w - 3, y - 12), (right_x + 3, y)], fill=(20, 20, 20))

    # 2. Torso (Safety Vest or Dark Shirt)
    torso_top = top_y + int(h * 0.22)
    torso_bottom = y - int(h * 0.45)
    if has_vest:
        # High-Vis Neon Orange / Yellow
        draw.rectangle([(left_x, torso_top), (right_x, torso_bottom)], fill=(255, 120, 10))
        # Reflective Silver Stripes
        stripe_y1 = torso_top + int((torso_bottom - torso_top) * 0.3)
        stripe_y2 = torso_top + int((torso_bottom - torso_top) * 0.7)
        draw.rectangle([(left_x, stripe_y1), (right_x, stripe_y1 + 5)], fill=(225, 235, 240))
        draw.rectangle([(left_x, stripe_y2), (right_x, stripe_y2 + 5)], fill=(225, 235, 240))
        draw.rectangle([(left_x + int(w * 0.35), torso_top), (left_x + int(w * 0.65), torso_bottom)], fill=(225, 235, 240))
    else:
        draw.rectangle([(left_x, torso_top), (right_x, torso_bottom)], fill=(40, 50, 70))

    # 3. Head & Face
    head_h = int(h * 0.20)
    head_w = int(w * 0.7)
    head_top = top_y + int(h * 0.05)
    head_left = x - head_w // 2
    head_right = x + head_w // 2
    draw.ellipse([(head_left, head_top), (head_right, head_top + head_h)], fill=(220, 180, 150))

    # 4. Hardhat
    if has_hardhat:
        # Yellow Safety Helmet
        hh_top = top_y
        hh_bottom = head_top + int(head_h * 0.55)
        hh_left = x - int(w * 0.48)
        hh_right = x + int(w * 0.48)
        draw.ellipse([(hh_left, hh_top), (hh_right, hh_bottom)], fill=(255, 215, 0), outline=(200, 160, 0), width=2)
        # Brim
        brim_y = hh_bottom - 4
        draw.rectangle([(hh_left - 4, brim_y), (hh_right + 4, brim_y + 4)], fill=(240, 195, 0))


def draw_heavy_machinery(draw: ImageDraw.Draw, x: int, y: int, scale: float = 1.0, is_reversing: bool = False):
    """
    Renders a heavy construction wheel loader / excavator.
    (x, y) is the bottom-center of the machine chassis.
    """
    w = int(240 * scale)
    h = int(180 * scale)
    top_y = y - h
    left_x = x - w // 2
    right_x = x + w // 2

    # 1. Heavy Rubber Tires / Tracks
    tire_w = int(50 * scale)
    tire_h = int(60 * scale)
    # Left wheel
    draw.rectangle([(left_x, y - tire_h), (left_x + tire_w, y)], fill=(35, 35, 35), outline=(15, 15, 15), width=3)
    draw.ellipse([(left_x + 5, y - tire_h + 5), (left_x + tire_w - 5, y - 5)], fill=(70, 70, 70))
    # Right wheel
    draw.rectangle([(right_x - tire_w, y - tire_h), (right_x, y)], fill=(35, 35, 35), outline=(15, 15, 15), width=3)
    draw.ellipse([(right_x - tire_w + 5, y - tire_h + 5), (right_x - 5, y - 5)], fill=(70, 70, 70))

    # 2. Main Chassis (CAT Yellow/Amber)
    body_left = left_x + int(tire_w * 0.4)
    body_right = right_x - int(tire_w * 0.4)
    body_top = y - int(h * 0.65)
    body_bottom = y - int(tire_h * 0.5)
    draw.rectangle([(body_left, body_top), (body_right, body_bottom)], fill=(235, 150, 10), outline=(40, 30, 10), width=3)

    # 3. Operator Cabin (Reinforced ROPS / Tinted Glass)
    cab_w = int(w * 0.42)
    cab_h = int(h * 0.45)
    cab_left = body_left + int(w * 0.1)
    cab_top = top_y
    draw.rectangle([(cab_left, cab_top), (cab_left + cab_w, body_top)], fill=(45, 55, 65), outline=(20, 20, 20), width=3)
    # Glass pane
    draw.rectangle([(cab_left + 6, cab_top + 6), (cab_left + cab_w - 6, body_top - 6)], fill=(120, 175, 205), outline=(30, 40, 50))

    # 4. Engine Hood / Counterweight
    hood_left = cab_left + cab_w
    draw.rectangle([(hood_left, body_top - int(cab_h * 0.6)), (body_right, body_top)], fill=(225, 140, 5), outline=(40, 30, 10), width=2)
    # Exhaust pipe
    draw.rectangle([(hood_left + 15, cab_top + 10), (hood_left + 22, body_top - int(cab_h * 0.6))], fill=(50, 50, 50))

    # 5. Loader Bucket / Articulated Arm
    bucket_w = int(65 * scale)
    bucket_h = int(45 * scale)
    b_left = left_x - bucket_w // 2
    b_right = left_x + bucket_w // 2
    draw.polygon([
        (b_left, y - 10),
        (b_right, y - 10),
        (b_right + 10, y - bucket_h),
        (b_left - 10, y - bucket_h)
    ], fill=(90, 95, 100), outline=(30, 30, 30))

    # 6. Reversing Alarm Beacon (Flashing Amber on top of Cab)
    beacon_color = (255, 230, 50) if is_reversing else (180, 100, 20)
    draw.rectangle([(cab_left + cab_w // 2 - 8, cab_top - 12), (cab_left + cab_w // 2 + 8, cab_top)], fill=beacon_color, outline=(100, 80, 10))


def render_scene_background(width: int = 1280, height: int = 720) -> Image.Image:
    """
    Renders an authentic industrial earthmoving site with excavation trench,
    compacted gravel tracks, safety barricades, and perimeter fencing.
    """
    img = Image.new("RGB", (width, height), color=(150, 135, 110))
    draw = ImageDraw.Draw(img)

    # 1. Sky & Horizon line
    horizon_y = int(height * 0.40)
    draw.rectangle([(0, 0), (width, horizon_y)], fill=(195, 210, 225))
    # Distant soil berms
    draw.rectangle([(0, horizon_y - 20), (width, horizon_y)], fill=(115, 100, 80))

    # 2. Compacted Gravel Roadway
    draw.polygon([
        (int(width * 0.15), horizon_y),
        (int(width * 0.85), horizon_y),
        (width, height),
        (0, height)
    ], fill=(130, 115, 90))

    # 3. Excavation Trench / Hazard Zone (Darker Pit)
    trench_y1 = int(height * 0.55)
    trench_y2 = int(height * 0.85)
    trench_x1 = int(width * 0.08)
    trench_x2 = int(width * 0.45)
    draw.rectangle([(trench_x1, trench_y1), (trench_x2, trench_y2)], fill=(65, 52, 40), outline=(45, 35, 25), width=3)
    draw.text((trench_x1 + 15, trench_y1 + 15), "EXCAVATION TRENCH - DANGER ZONE", fill=(210, 160, 60))

    # 4. Trench Warning Barricades
    for bx in range(trench_x1, trench_x2, 80):
        draw.rectangle([(bx, trench_y1 - 25), (bx + 70, trench_y1 - 5)], fill=(255, 90, 20), outline=(255, 255, 255), width=2)
        draw.line([(bx, trench_y1 - 5), (bx, trench_y1)], fill=(50, 50, 50), width=4)
        draw.line([(bx + 70, trench_y1 - 5), (bx + 70, trench_y1)], fill=(50, 50, 50), width=4)

    # 5. Tire Track Marks
    for offset in range(-60, 60, 25):
        draw.line([(int(width * 0.55) + offset, horizon_y), (int(width * 0.70) + offset * 2, height)], fill=(110, 95, 75), width=4)

    return img


def generate_scenario_1_near_miss(output_path: str, num_frames: int = 120, fps: int = 20):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    bg_base = render_scene_background(width, height)

    for f in range(num_frames):
        frame_img = bg_base.copy()
        draw = ImageDraw.Draw(frame_img)

        t = f / float(num_frames)
        w_x = int(220 + t * 460)
        w_y = int(560 + t * 60)
        w_scale = 0.85 + (w_y / 720.0) * 0.35
        draw_realistic_worker(draw, w_x, w_y, scale=w_scale, has_hardhat=True, has_vest=True)

        m_x = int(1050 - t * 440)
        m_y = int(550 + t * 80)
        m_scale = 0.85 + (m_y / 720.0) * 0.35
        is_rev_blink = (f % 6 < 3)
        draw_heavy_machinery(draw, m_x, m_y, scale=m_scale, is_reversing=is_rev_blink)

        cv_frame = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_frame, f"CAM-02-MAST // UNSEEN TEST SEQUENCE // FRAME {f:04d}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        out.write(cv_frame)

    out.release()
    print(f"Generated Scenario 1: {output_path}")


def generate_scenario_2_safe_coworking(output_path: str, num_frames: int = 120, fps: int = 20):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    bg_base = render_scene_background(width, height)

    for f in range(num_frames):
        frame_img = bg_base.copy()
        draw = ImageDraw.Draw(frame_img)

        draw.line([(350, 480), (450, 720)], fill=(255, 255, 255), width=3)

        t = f / float(num_frames)
        w_x = int(380 + t * 60)
        w_y = int(520 + t * 160)
        w_scale = 0.85 + (w_y / 720.0) * 0.35
        draw_realistic_worker(draw, w_x, w_y, scale=w_scale, has_hardhat=True, has_vest=True)

        m_x = int(720 + t * 70)
        m_y = int(520 + t * 160)
        m_scale = 0.85 + (m_y / 720.0) * 0.35
        draw_heavy_machinery(draw, m_x, m_y, scale=m_scale, is_reversing=False)

        cv_frame = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_frame, f"CAM-02-MAST // CONTROLLED PARALLEL WORK // FRAME {f:04d}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        out.write(cv_frame)

    out.release()
    print(f"Generated Scenario 2: {output_path}")


def generate_scenario_3_trench_breach(output_path: str, num_frames: int = 120, fps: int = 20):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    bg_base = render_scene_background(width, height)

    for f in range(num_frames):
        frame_img = bg_base.copy()
        draw = ImageDraw.Draw(frame_img)

        t = f / float(num_frames)
        w_x = int(550 - t * 240)
        w_y = int(500 + t * 120)
        w_scale = 0.85 + (w_y / 720.0) * 0.35
        draw_realistic_worker(draw, w_x, w_y, scale=w_scale, has_hardhat=False, has_vest=True)

        m_x = 340
        m_y = 560
        m_scale = 0.95
        draw_heavy_machinery(draw, m_x, m_y, scale=m_scale, is_reversing=(f % 10 < 5))

        cv_frame = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_frame, f"CAM-02-MAST // DYNAMIC TRENCH BREACH & PPE VIOLATION // FRAME {f:04d}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        out.write(cv_frame)

    out.release()
    print(f"Generated Scenario 3: {output_path}")


def generate_all_demo_videos(target_dir: str = "data/test_videos"):
    dest = Path(target_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    generate_scenario_1_near_miss(str(dest / "scenario_1_near_miss.mp4"), num_frames=120, fps=20)
    generate_scenario_2_safe_coworking(str(dest / "scenario_2_safe_coworking.mp4"), num_frames=120, fps=20)
    generate_scenario_3_trench_breach(str(dest / "scenario_3_trench_breach.mp4"), num_frames=120, fps=20)
    print("All demo video sequences successfully generated in:", dest)


if __name__ == "__main__":
    generate_all_demo_videos()
