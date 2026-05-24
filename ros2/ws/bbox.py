import struct, glob, os


def bbox(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for _ in range(n):
            f.read(12)
            for _ in range(3):
                v = struct.unpack("<fff", f.read(12))
                for i in range(3):
                    if v[i] < mins[i]:
                        mins[i] = v[i]
                    if v[i] > maxs[i]:
                        maxs[i] = v[i]
            f.read(2)
    return mins, maxs, n


for fn in sorted(glob.glob("/ws/src/piper_ros/src/piper_description/meshes/*.STL")):
    name = os.path.basename(fn)
    mins, maxs, n = bbox(fn)
    extent = [maxs[i] - mins[i] for i in range(3)]
    print(
        f"{name:<20}  tri={n:>5}  "
        f"X[{mins[0]:+.4f}..{maxs[0]:+.4f}]={extent[0]:.4f}  "
        f"Y[{mins[1]:+.4f}..{maxs[1]:+.4f}]={extent[1]:.4f}  "
        f"Z[{mins[2]:+.4f}..{maxs[2]:+.4f}]={extent[2]:.4f}"
    )
