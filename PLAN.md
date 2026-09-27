# Kurdish OCR fine-tuning plan

## Decision

This project is a **text recognition** project, not a text detection project:
the dataset contains cropped text images and one UTF-8 transcription per image.
Do not create detection boxes or use a detection config for this dataset.

Start with **`arabic_PP-OCRv5_mobile_rec`**. Sorani and Arabic-script Badini use
the Arabic script, and this is the closest official PaddleOCR recognition base
model. It is small enough to export and deploy (about 7.6 MB as an inference
model) while retaining the Arabic-script feature extractor. Fine-tuning will
replace or extend the character dictionary with the characters actually present
in this dataset.

The newer **PP-OCRv6** pipeline advertises `ku` support and should be used as a
baseline comparison later. It is not the first training choice here because the
dataset explicitly includes Sorani/Badini Arabic-script text and PP-OCRv5 has
an explicit Arabic recognition model.

Dataset: https://huggingface.co/datasets/a7x3a/qai-ocr-v1-small

## What the dataset contains

The Hugging Face dataset card reports:

- `train`, `val`, and `test` splits.
- WebDataset TAR shards, with up to 5,000 samples per shard.
- Each sample has matching files such as `000000000000.jpg` and
  `000000000000.txt`.
- The TXT label contains only the transcription and is UTF-8 encoded.
- Languages/tags include Kurdish, Sorani, Badini, Arabic, and English.

The repository is about 2.9 GB according to the Hub API. Confirm the actual
sample count after download rather than assuming that the number of TAR files
equals the number of examples.

## 1. Prepare a reproducible environment

Use Linux, WSL2, or a CUDA-enabled training machine for PaddleOCR training.
Windows can be used for inspection and inference, but Paddle/PaddleOCR CUDA
support is generally easier to reproduce in WSL2 or Linux.

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Install the PaddlePaddle build that matches the machine's CUDA version using
the official PaddlePaddle installation selector. Then verify:

```bash
python -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

Pin the PaddleOCR commit, PaddlePaddle version, CUDA version, and Python
version in the experiment notes. Do not mix a PaddleOCR 2.x training command
with a 3.x checkout.

## 2. Download and inspect the dataset

The current Hub CLI is `hf`:

```bash
hf download a7x3a/qai-ocr-v1-small --repo-type dataset --local-dir data/qai-ocr-v1-small
```

Before converting anything, inspect `metadata.jsonl` and count matching image
and text members in every shard. Reject samples when:

- the image or label is missing;
- the label is empty when empty labels are not intentional;
- the image cannot be decoded;
- the label contains a newline or unexpected control character;
- the label has leading/trailing spaces that are not part of the ground truth.

Do not normalize Arabic/Kurdish text blindly. Preserve the ground truth as
written. Only apply a documented normalization policy after measuring how many
labels it changes. In particular, do not remove diacritics, punctuation,
spaces, or Kurdish-specific characters just to make training easier.

## 3. Convert WebDataset shards to PaddleOCR format

PaddleOCR recognition training expects one line per sample:

```text
relative/path/to/image.jpg<TAB>transcription
```

Create this layout after extracting the TAR members:

```text
data/kurdish_rec/
  images/
    train/...
    val/...
    test/...
  train_rec.txt
  val_rec.txt
  test_rec.txt
  kurdish_dict.txt
```

Write paths relative to `data/kurdish_rec`, use a literal tab separator, and
write all annotation files as UTF-8. The conversion script must be deterministic
and must preserve the official split; never randomly mix `test` samples into
training.

Generate `kurdish_dict.txt` from the training labels, one unique character per
line, after deciding how spaces are represented. Include every character found
in `val` and `test` too, or report those as out-of-vocabulary errors before
training. The dictionary must cover Kurdish Arabic-script letters, Arabic and
Persian-compatible letters, Latin letters, digits, punctuation, and the space
character if present.

Run a data audit before training:

```bash
python tools/check_rec_dataset.py --root data/kurdish_rec
```

If the repository does not contain that checker, create a small local checker
that validates image existence, image decoding, UTF-8 labels, tab separation,
duplicate paths, empty labels, maximum label length, and dictionary coverage.
Keep its report with the experiment.

## 4. Configure recognition training

Start from the Arabic multilingual recognition config shipped with the checked
out PaddleOCR version:

```text
configs/rec/multi_language/rec_arabic_lite_train.yml
```

Copy it to `configs/rec/custom/kurdish_rec.yml` and change only the dataset,
dictionary, batch, output, and pretrained-weight settings. Keep the model
architecture and preprocessing compatible with the selected Arabic checkpoint.
The important settings are conceptually:

```yaml
Global:
  character_dict_path: ./data/kurdish_rec/kurdish_dict.txt
  max_text_length: 64
  use_space_char: true
  pretrained_model: ./pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained

Train:
  dataset:
    data_dir: ./data/kurdish_rec
    label_file_list:
      - ./data/kurdish_rec/train_rec.txt

Eval:
  dataset:
    data_dir: ./data/kurdish_rec
    label_file_list:
      - ./data/kurdish_rec/val_rec.txt
```

Use the exact key names from the copied config; PaddleOCR config schemas vary
between releases. Download the official **training** checkpoint for
`arabic_PP-OCRv5_mobile_rec`, not only its inference TAR. If that checkpoint
does not match the copied config, stop and use the matching official config
generated by that PaddleOCR release.

Recommended initial settings:

- learning rate: start around `0.0005` and lower it if validation CER worsens;
- batch size: largest value that fits GPU memory, starting at 64 for a small
  mobile model when possible;
- epochs: 30-100, with early stopping based on validation CER;
- save the best checkpoint by validation accuracy/CER, not merely the final
  epoch;
- enable mixed precision only after a short FP32 smoke test succeeds.

Run a smoke test first with a small sample and one epoch. Then run the full
training job:

```bash
python tools/train.py \
  -c configs/rec/custom/kurdish_rec.yml \
  -o Global.save_model_dir=./output/kurdish_ppocrv5_arabic \
     Global.pretrained_model=./pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained
```

Use `-o` overrides only for keys that exist in the checked-out config. Keep the
YAML file as the authoritative experiment configuration.

## 5. Evaluate Kurdish OCR properly

Evaluate the untouched `test` split only after model selection. Report at least:

- character error rate (CER);
- word error rate (WER), using a documented whitespace policy;
- exact-match accuracy;
- results separately for Sorani, Badini, Arabic, and English when metadata
  permits it;
- results by text length, image width/height, font, blur, and difficult glyphs.

Compare three checkpoints on the same test set:

1. The unfine-tuned Arabic PP-OCRv5 checkpoint.
2. The fine-tuned Kurdish checkpoint.
3. A PP-OCRv6 `ku` baseline, if the installed release provides its matching
   recognition weights and config.

Select the model by test CER/WER and deployment requirements, not by training
loss. Keep a fixed set of manually reviewed examples to catch visually
plausible but linguistically incorrect predictions.

## 6. Export the selected model

Export the best training checkpoint to inference format using the export command
from the checked-out PaddleOCR release. In the classic PaddleOCR 2.x/3.x
training workflow this is typically:

```bash
python tools/export_model.py \
  -c configs/rec/custom/kurdish_rec.yml \
  -o Global.pretrained_model=./output/kurdish_ppocrv5_arabic/best_accuracy \
     Global.save_inference_dir=./export/kurdish_ppocrv5_arabic
```

Verify that the export directory contains the inference model files and the
same character dictionary used during training. Never deploy a model with a
different dictionary order.

Run a local inference regression test on unseen images and compare predictions
with the Python training evaluator. Record model size, latency, memory usage,
and the Paddle/PaddleOCR versions.

For the current PaddleOCR 3.x pipeline, load the exported recognition model by
setting `text_recognition_model_dir` in the OCR pipeline configuration and
disable unnecessary modules for cropped images:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_recognition_model_dir="./export/kurdish_ppocrv5_arabic",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
```

If the application receives full pages later, this recognition model still
needs a separate text detector. Recognition fine-tuning does not teach the
model where text is located.

## 7. Publish and maintain the model

Export the selected model as a versioned artifact containing:

- inference weights;
- `kurdish_dict.txt`;
- the exact YAML config;
- dataset revision and SHA;
- training and evaluation metrics;
- supported scripts/languages and known limitations;
- an inference example and license/attribution information.

Upload it to a private Hugging Face model repository first, validate download
and inference from a clean environment, then make it public if appropriate:

```bash
hf repos create a7x3a/qai-ocr-kurdish --type model --private
hf upload a7x3a/qai-ocr-kurdish ./export/kurdish_ppocrv5_arabic --include "*"
```

## Acceptance criteria

The first release is ready only when:

- every train/val/test sample passes the data audit;
- the model beats the untouched Arabic checkpoint on Kurdish validation data;
- the test set was not used for tuning;
- CER, WER, and exact-match scores are recorded;
- exported inference predictions match training-checkpoint predictions;
- the model loads successfully from a clean environment using the documented
  dictionary and config;
- the artifact can be downloaded and run without the training source tree.
