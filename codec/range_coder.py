TOP = 1 << 24
BOT = 1 << 16
MASK = 0xFFFFFFFF

class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.rng = MASK
        self.out = bytearray()

    def encode(self, cum, freq, tot):
        r = self.rng // tot
        self.low = (self.low + r * cum) & MASK
        self.rng = r * freq
        self._renorm()

    def _renorm(self):
        while True:
            if (self.low ^ (self.low + self.rng)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK

    def finish(self):
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK
        return bytes(self.out)

class RangeDecoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.low = 0
        self.rng = MASK
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & MASK
        self._r = 0

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def get_freq(self, tot):
        self._r = self.rng // tot
        v = ((self.code - self.low) & MASK) // self._r
        return min(v, tot - 1)

    def decode(self, cum, freq, tot):
        r = self._r
        self.low = (self.low + r * cum) & MASK
        self.rng = r * freq
        while True:
            if (self.low ^ (self.low + self.rng)) < TOP:
                pass
            elif self.rng < BOT:
                self.rng = (-self.low) & (BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._byte()) & MASK
            self.low = (self.low << 8) & MASK
            self.rng = (self.rng << 8) & MASK
