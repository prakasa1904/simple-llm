"""Tokenizer per huruf (character-level).

Model besar memakai tokenizer sub-kata (misalnya "belajar" -> ["bel", "ajar"]).
Di sini setiap karakter adalah satu token, supaya mudah dipahami:

    "Kancil" -> ['K', 'a', 'n', 'c', 'i', 'l'] -> [23, 31, 44, 33, 39, 42]
"""


class CharTokenizer:
    def __init__(self, karakter):
        # Daftar semua karakter yang dikenal. Posisi dalam daftar = ID token.
        self.karakter = list(karakter)
        self.ke_id = {c: i for i, c in enumerate(self.karakter)}

    @classmethod
    def dari_teks(cls, teks):
        """Bangun kosakata dari semua karakter unik di dalam teks."""
        return cls(sorted(set(teks)))

    @property
    def ukuran_kosakata(self):
        return len(self.karakter)

    def encode(self, teks):
        """Teks -> daftar ID. Karakter yang tidak dikenal dilewati."""
        return [self.ke_id[c] for c in teks if c in self.ke_id]

    def decode(self, ids):
        """Daftar ID -> teks."""
        return "".join(self.karakter[i] for i in ids)

    def karakter_asing(self, teks):
        """Karakter di dalam teks yang tidak ada di kosakata."""
        return sorted({c for c in teks if c not in self.ke_id})
