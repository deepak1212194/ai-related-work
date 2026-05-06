# 04 · CLIP Image-Text Search

A small **multimodal retrieval** demo: given a folder of images and a natural-language query, return the images that best match the query — and vice versa.

Powered by OpenAI's **CLIP** (`openai/clip-vit-base-patch32`). Image embeddings are pre-computed once and cached; queries are encoded on the fly.

## Why this design

Multimodal search shows up in any product that handles user-generated images: blueprint review, document understanding, content moderation, e-commerce. The pattern is always the same — embed images and queries into the same vector space, then do nearest-neighbour search. CLIP is the simplest way to get there.

## Quick start

```bash
cd 04-clip-image-text-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Drop a few JPEGs/PNGs into ./data/images/
python -m src.embed_images       # builds the image-vector cache
python -m src.search --query "a photo of a dog on a beach" --top-k 3
```

## Project layout

```
04-clip-image-text-search/
├── src/
│   ├── embed_images.py   # Module 1: load CLIP, encode images, cache vectors
│   └── search.py         # Module 2: encode the query, return top-K matches
├── data/
│   └── images/           # drop your own images here
└── requirements.txt
```

## Notes

- All sample images you place in `data/images/` stay local — nothing is uploaded.
- Default model is `clip-vit-base-patch32` for speed; swap to `clip-vit-large-patch14` for better quality.
- The image cache is a simple `.npz` file plus a JSON manifest; no vector DB needed for under a few thousand images.
