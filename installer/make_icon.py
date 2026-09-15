"""Generate the JARVIS arc-reactor icon as a multi-size .ico, with no image library."""
import math, struct, zlib

def png_bytes(size):
    """Render the arc reactor at `size` px and return PNG bytes (RGBA)."""
    c = (size - 1) / 2.0
    px = bytearray()
    # colours
    CORE   = (215, 245, 255)
    RING   = (41, 182, 246)
    GLOW   = (27, 118, 170)
    DARK   = (10, 24, 36)

    def blend(a, b, t):
        return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            dx, dy = x - c, y - c
            r = math.hypot(dx, dy) / c          # 0 at centre, 1 at edge
            a = 0
            col = DARK
            if r <= 1.0:
                if r < 0.20:                     # bright core
                    col = blend(CORE, (255, 255, 255), max(0.0, 1 - r / 0.20) * 0.5)
                    a = 255
                elif r < 0.30:                   # core falloff into the dark well
                    t = (r - 0.20) / 0.10
                    col = blend(CORE, DARK, t)
                    a = 255
                elif r < 0.44:                   # dark gap
                    col = DARK
                    a = 235
                elif r < 0.52:                   # inner bright ring
                    t = abs(r - 0.48) / 0.04
                    col = blend(RING, DARK, t * 0.8)
                    a = 255
                elif r < 0.62:                   # dark gap
                    col = DARK
                    a = 235
                elif r < 0.78:                   # segmented outer ring (eight coils)
                    ang = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)
                    seg = (ang * 8.0) % 1.0
                    lit = 0.18 < seg < 0.82
                    edge = abs(r - 0.70) / 0.08
                    if lit:
                        col = blend(RING, GLOW, edge)
                        a = 255
                    else:
                        col = DARK
                        a = 245
                elif r < 0.86:                   # outer rim
                    t = abs(r - 0.82) / 0.04
                    col = blend(GLOW, DARK, t)
                    a = 255
                else:                            # soft glow to the edge
                    t = (r - 0.86) / 0.14
                    col = blend(GLOW, DARK, t)
                    a = int(255 * (1 - t) ** 1.5)
            row += bytes((col[0], col[1], col[2], a))
        rows.append(bytes(row))

    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


SIZES = [16, 24, 32, 48, 64, 128, 256]
images = [(s, png_bytes(s)) for s in SIZES]

header = struct.pack("<HHH", 0, 1, len(images))
offset = 6 + 16 * len(images)
entries, blobs = b"", b""
for size, data in images:
    w = 0 if size >= 256 else size
    entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(data), offset)
    blobs += data
    offset += len(data)

with open("../assets/jarvis.ico", "wb") as fh:
    fh.write(header + entries + blobs)
print(f"assets/jarvis.ico written: {len(header + entries + blobs)} bytes, sizes {SIZES}")

# A separate installer icon: same reactor, warmer ring, so the two are distinguishable.
