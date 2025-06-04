class Cube:
    def __init__(self, x, y, z, quat):
        self.x = x
        self.y = y
        self.z = z
        self.quat = quat  # [qx, qy, qz, qw]

    def __str__(self):
        return f"Cube(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, quat={self.quat})"