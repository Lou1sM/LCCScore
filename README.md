Code for the paper [Local Composition Complexity: How to Detect a Human-Readable Message](https://arxiv.org/abs/2501.03664).

This paper introduces the LCC score, which is a computable, information-theoretic metric that is high when data is meaningfully structured. It gives a high score to real-world images, natural language text and spoken natural language, across multiple languages, and a low score to uniform/repetitive data and to random data.

To reproduce the experiments in the paper on images, audio and text, using `image_complexity.py`, `audio_complexity.py` and `text_complexity.py`. To apply the LCC score to your own data, see `example_usage_image.py`, `example_usage_audio.py`, and `example_usage_text.py`. 

If the theory or implementation of the LCC score is useful in your work, please cite it.

```
@article{mahon2025local,
  title={Local Compositional Complexity: How to Detect a Human-readable Messsage},
  author={Mahon, Louis},
  journal={arXiv preprint arXiv:2501.03664},
  year={2025}
}
```

If there are any difficulties running the code, please open an issue or contact me. Thanks :)
