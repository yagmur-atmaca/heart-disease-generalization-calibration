# -*- coding: utf-8 -*-
"""Projeyi ana betikle çalıştırdıktan sonra sayısal çıktıların doğruluğunu kontrol eder."""
from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Faz1Kontrol(unittest.TestCase):
    def test_dogrulama_ozeti(self):
        p=ROOT/'ciktilar'/'faz1_dogrulama_ozeti.json'
        self.assertTrue(p.exists(), 'Önce python faz1_analiz.py çalıştırın')
        d=json.loads(p.read_text())
        self.assertEqual(d['farkli_tam_kayit'],596)
        self.assertEqual(d['hungarian_eksik'],778)
        self.assertEqual(d['hungarian_ortak_10_eksik'],34)
        self.assertEqual(d['ortak_egitim_matrisi'],[242,18])
        self.assertEqual(d['hungarian_donusum_matrisi'],[293,18])
    def test_gercek_pngler(self):
        self.assertEqual(len(list((ROOT/'grafikler').glob('*.png'))),7)
if __name__=='__main__':
    unittest.main()
