from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report"
OUT_PATH = OUT_DIR / "architecture_diagram.png"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\seguisb.ttf",
                r"C:\Windows\Fonts\msjhbd.ttc",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\msjh.ttc",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
BOX_TITLE_FONT = load_font(22, bold=True)
TEXT_FONT = load_font(18)
SMALL_FONT = load_font(16)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if width <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_box(draw, xy, title, body_lines, fill, outline="#1f2937"):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 40), radius=18, fill=outline)
    draw.rectangle((x1, y1 + 20, x2, y1 + 40), fill=outline)
    draw.text((x1 + 16, y1 + 9), title, font=BOX_TITLE_FONT, fill="white")
    y = y1 + 54
    for line in body_lines:
        draw.text((x1 + 16, y), line, font=TEXT_FONT, fill="#111827")
        y += 26


def arrow(draw, start, end, text=None, text_offset=(0, 0), color="#374151", width=4):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        tip = (ex, ey)
        wing1 = (ex - 14 * sign, ey - 8)
        wing2 = (ex - 14 * sign, ey + 8)
    else:
        sign = 1 if dy >= 0 else -1
        tip = (ex, ey)
        wing1 = (ex - 8, ey - 14 * sign)
        wing2 = (ex + 8, ey - 14 * sign)
    draw.polygon([tip, wing1, wing2], fill=color)
    if text:
        mx = (sx + ex) // 2 + text_offset[0]
        my = (sy + ey) // 2 + text_offset[1]
        bbox = draw.textbbox((0, 0), text, font=SMALL_FONT)
        pad = 6
        draw.rounded_rectangle(
            (mx - pad, my - pad, mx + (bbox[2] - bbox[0]) + pad, my + (bbox[3] - bbox[1]) + pad),
            radius=8,
            fill="white",
            outline="#d1d5db",
        )
        draw.text((mx, my), text, font=SMALL_FONT, fill="#111827")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1800, 1100), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((20, 20, 1780, 1080), radius=28, outline="#cbd5e1", width=3, fill="#f8fafc")
    draw.text((55, 40), "PPO for 2D Lattice Protein Folding", font=TITLE_FONT, fill="#0f172a")
    draw.text((58, 88), "Actor-Critic architecture, environment interaction, and reward design", font=TEXT_FONT, fill="#475569")

    boxes = {
        "env": (60, 170, 430, 430),
        "state": (510, 170, 900, 430),
        "encoder": (980, 170, 1330, 430),
        "actor": (1410, 150, 1730, 360),
        "critic": (1410, 390, 1730, 600),
        "buffer": (980, 520, 1330, 820),
        "gae": (560, 530, 900, 790),
        "reward": (60, 520, 470, 880),
    }

    draw_box(
        draw,
        boxes["env"],
        "Environment",
        [
            "2D lattice HP folding",
            "Sequence: H / P residues",
            "Action space: left, straight, right",
            "Checks collision and valid placement",
            "(s_{t+1}, r_t, done) = Env(s_t, a_t)",
        ],
        "#dbeafe",
    )
    draw_box(
        draw,
        boxes["state"],
        "State s_t",
        [
            "Grid occupancy tensor G_t",
            "Current endpoint p_t",
            "Current direction d_t",
            "Residue index i_t",
            "HH contact features c_t",
        ],
        "#dcfce7",
    )
    draw_box(
        draw,
        boxes["encoder"],
        "Shared Encoder",
        [
            "MLP or CNN + MLP",
            "Extract folding features",
            "Hidden layers: 128 -> 128",
        ],
        "#fef3c7",
    )
    draw_box(
        draw,
        boxes["actor"],
        "Actor",
        [
            "Policy pi_theta(a_t | s_t)",
            "Softmax(f_theta(s_t))",
            "Sample action a_t",
        ],
        "#fee2e2",
    )
    draw_box(
        draw,
        boxes["critic"],
        "Critic",
        [
            "Value V_phi(s_t)",
            "Estimates expected return",
            "V_phi(s_t) ~ E[sum gamma^k r_{t+k}]",
        ],
        "#ede9fe",
    )
    draw_box(
        draw,
        boxes["buffer"],
        "Trajectory Buffer",
        [
            "Store: s_t, a_t, r_t",
            "log pi_old(a_t | s_t)",
            "V(s_t), done flags",
            "Rollout batch for PPO",
        ],
        "#e0f2fe",
    )
    draw_box(
        draw,
        boxes["gae"],
        "GAE + PPO Update",
        [
            "delta_t = r_t + gamma V(s_{t+1}) - V(s_t)",
            "Advantage A_hat_t with GAE",
            "Actor: clipped PPO objective",
            "Critic: MSE(V(s_t), R_hat_t)",
        ],
        "#fae8ff",
    )
    draw_box(
        draw,
        boxes["reward"],
        "Reward Design",
        [
            "r_t = valid + contact + compactness + terminal",
            "+0.1 valid placement",
            "-1.0 invalid / self-overlap",
            "+0.5 Delta HH contact",
            "-0.01 Delta bounding-box area",
            "Final bonus: 2.0 * N_HH_final",
        ],
        "#ffedd5",
    )

    arrow(draw, (430, 300), (510, 300), "state output", (0, -26))
    arrow(draw, (900, 300), (980, 300), "feature input", (0, -26))
    arrow(draw, (1330, 240), (1410, 240), "policy head", (0, -26))
    arrow(draw, (1330, 480), (1410, 480), "value head", (0, -26))
    arrow(draw, (1570, 360), (1570, 450), "V_phi(s_t)", (16, 0))
    arrow(draw, (1570, 255), (1570, 150), "a_t ~ pi_theta", (16, -30))
    arrow(draw, (1570, 150), (1570, 120))
    arrow(draw, (1570, 120), (245, 120))
    arrow(draw, (245, 120), (245, 170), "action to env", (-10, -20))
    arrow(draw, (430, 380), (980, 670), "rollout data", (0, -20))
    arrow(draw, (1730, 500), (1330, 670), "value estimates", (0, -22))
    arrow(draw, (980, 690), (900, 690), "compute A_hat_t, R_hat_t", (-20, -24))
    arrow(draw, (560, 660), (470, 660), "reward terms", (-10, -24))
    arrow(draw, (730, 790), (730, 930))
    arrow(draw, (730, 930), (1570, 930))
    arrow(draw, (1570, 930), (1570, 600), "update actor / critic", (15, 0))

    ppo_formula = "L_actor = E[min(r_t(theta) A_hat_t, clip(r_t(theta), 1-e, 1+e) A_hat_t)]"
    critic_formula = "L_critic = E[(V_phi(s_t) - R_hat_t)^2]"
    draw.rounded_rectangle((980, 860, 1730, 1030), radius=20, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((1005, 885), ppo_formula, font=SMALL_FONT, fill="#111827")
    draw.text((1005, 925), critic_formula, font=SMALL_FONT, fill="#111827")
    draw.text((1005, 970), "Interaction loop: s_t -> actor -> a_t -> environment -> r_t, s_{t+1} -> PPO update", font=SMALL_FONT, fill="#334155")

    image.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
