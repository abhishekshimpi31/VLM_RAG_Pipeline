import os
import sys
import unittest

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.common.file_versioning import get_safe_chapter_id, get_file_hash

try:
    from src.ingestion.pdf_image_rendering import clean_markdown_text, extract_figure_captions
except ImportError:
    clean_markdown_text = None
    extract_figure_captions = None

try:
    from src.evaluation.evaluate_retrieval import compute_ir_metrics
except ImportError:
    compute_ir_metrics = None


class TestPipelineSanity(unittest.TestCase):

    def test_config_paths(self):
        """Verify configuration paths are set properly."""
        self.assertTrue(hasattr(config, "BASE_DIR"))
        self.assertTrue(hasattr(config, "COLLECTION_NAME"))
        self.assertEqual(config.COLLECTION_NAME, "ipcc_hybrid_chunks")

    def test_safe_chapter_id(self):
        """Verify hashing generates safe directory IDs."""
        cid = get_safe_chapter_id("Chapter 4: Future Global Climate")
        self.assertTrue(cid.startswith("chap_"))
        self.assertEqual(len(cid), 13)

    def test_clean_markdown_text(self):
        """Verify markdown syntax stripping."""
        if clean_markdown_text is None:
            self.skipTest("pymupdf not installed in current interpreter")
        raw = "### Chapter 1\n**Bold Statement** and <br> table | col"
        cleaned = clean_markdown_text(raw)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("<br>", cleaned)

    def test_figure_caption_extraction(self):
        """Verify figure caption regex parsing."""
        if extract_figure_captions is None:
            self.skipTest("pymupdf not installed in current interpreter")
        sample_page = "Some intro text.\n\n**Figure 3.12** | Global temperature anomaly projection.\n\nMore text."
        captions = extract_figure_captions(sample_page)
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0]["figure_id"], "Figure 3.12")

    def test_ir_metrics_computation(self):
        """Verify IR metrics calculate precision, recall, MRR, NDCG accurately."""
        if compute_ir_metrics is None:
            self.skipTest("pandas not installed in current interpreter")
        rel = [1, 0, 0]
        metrics = compute_ir_metrics(rel)
        self.assertAlmostEqual(metrics["precision_at_k"], 0.3333, places=3)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()

