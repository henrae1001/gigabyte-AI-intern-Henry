from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gigabyte_rag.chunking import build_chunks
from gigabyte_rag.llm import build_llama_cli_command
from gigabyte_rag.parser import parse_specs, seed_specs, validate_specs
from gigabyte_rag.pipeline import retrieve
from gigabyte_rag.vector_index import HashingVectorIndex


class ParserValidationTests(unittest.TestCase):
    def test_validate_specs_rejects_missing_required_section(self) -> None:
        specs = seed_specs()
        first_model = specs["models"][0]
        first_model["sections"] = [
            section for section in first_model["sections"] if section["section"] != "顯示晶片"
        ]

        with self.assertRaisesRegex(ValueError, "missing sections"):
            validate_specs(specs)

    def test_parse_specs_reads_variant_values_from_html_table(self) -> None:
        html = """
        <table>
          <tr><th>Spec</th><th>AORUS MASTER 16 BXH</th><th>AORUS MASTER 16 BYH</th><th>AORUS MASTER 16 BZH</th></tr>
          <tr><th>GPU</th><td>RTX 5090 Laptop GPU; 24GB GDDR7</td><td>RTX 5080 Laptop GPU; 16GB GDDR7</td><td>RTX 5070 Ti Laptop GPU; 12GB GDDR7</td></tr>
          <tr><th>Battery</th><td>99Wh</td><td>99Wh</td><td>99Wh</td></tr>
          <tr><th>Adapter</th><td>330W AC Adapter</td><td>330W AC Adapter</td><td>330W AC Adapter</td></tr>
          <tr><th>Weight</th><td>2.5 kg</td><td>2.5 kg</td><td>2.5 kg</td></tr>
        </table>
        """
        specs = parse_specs(html)
        first_model = specs["models"][0]
        sections = {section["section"]: section["values"] for section in first_model["sections"]}

        self.assertIn("RTX 5090 Laptop GPU", sections["顯示晶片"][0])
        self.assertEqual(sections["電池"], ["99Wh"])
        self.assertEqual(sections["變壓器"], ["330W AC Adapter"])


class RetrievalTests(unittest.TestCase):
    def test_retrieves_model_specific_gpu_chunk(self) -> None:
        index_path = _build_temp_index()

        results, metrics = retrieve(
            "AORUS MASTER 16 BXH 的 GPU 規格是什麼？",
            index_path=index_path,
            model_filter="BXH",
        )

        self.assertGreater(metrics.seconds, 0)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.id, "bxh-顯示晶片")

    def test_refuses_out_of_scope_price_question(self) -> None:
        index_path = _build_temp_index()

        results, _ = retrieve(
            "Does the official spec mention the laptop price?",
            index_path=index_path,
        )

        self.assertEqual(results, [])

    def test_aliases_retrieve_display_and_adapter_chunks(self) -> None:
        index_path = _build_temp_index()

        display_results, _ = retrieve("What is the resolution and refresh rate?", index_path=index_path)
        adapter_results, _ = retrieve("充電器 wattage 是多少？", index_path=index_path)

        self.assertTrue(display_results)
        self.assertIn("顯示器", display_results[0].chunk.section)
        self.assertTrue(adapter_results)
        self.assertIn("變壓器", adapter_results[0].chunk.section)


class LlamaCliTests(unittest.TestCase):
    def test_build_llama_cli_command_keeps_prompt_as_single_argument(self) -> None:
        command = build_llama_cli_command(
            "llama-cli.exe",
            "models/model.gguf",
            "Question with spaces",
            n_ctx=2048,
            n_gpu_layers=0,
            max_tokens=96,
            temperature=0.1,
        )

        self.assertEqual(command[0], "llama-cli.exe")
        self.assertIn("--no-display-prompt", command)
        self.assertEqual(command[command.index("-p") + 1], "Question with spaces")
        self.assertEqual(command[command.index("-ngl") + 1], "0")


def _build_temp_index() -> Path:
    chunks = build_chunks(seed_specs())
    temp_dir = tempfile.TemporaryDirectory()
    index_path = Path(temp_dir.name) / "vectors.json"
    index = HashingVectorIndex.build(chunks)
    index.save(index_path)
    _TEMP_DIRS.append(temp_dir)
    return index_path


_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


if __name__ == "__main__":
    unittest.main()
