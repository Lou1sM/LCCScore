import numpy as np
from .utils import opt_cost_from_counts, opt_cost_from_discrete_seq


def sorted_tok_counts(toks):
    pairs, counts = np.unique(toks, return_counts=True)
    assert sum(counts)==len(toks)
    tokcounts = sorted(zip(pairs, counts), key=lambda x:x[1])
    return tokcounts

def discrete_lcc(text, nits=1):
    orig_vocab, orig_counts = np.unique(list(text), return_counts=True)
    best_cost = opt_cost_from_counts(orig_counts)
    codebook = {}
    chunk_size = 60
    chunks = [text[i*chunk_size:(i+1)*chunk_size] for i in range(len(text)//chunk_size)]
    np.random.shuffle(chunks)
    available_unicode_points = [i for i in reversed(range(10000)) if chr(i) not in text]
    for it in range(nits):
        maybe_words = []
        for n in range(2,10):
            new_ngrams = [''.join(text[i:i+n]) for i in range(len(text)-n+1)]
            maybe_words += sorted_tok_counts(new_ngrams)[-20:]
        maybe_words.sort(key=lambda x:len(x[0]))
        maybe_words.sort(key=lambda x:x[1])
        for mw,_ in reversed(maybe_words):
            new_unicode_point = available_unicode_points[0]
            maybe_new_text = text.replace(mw, chr(new_unicode_point))
            new_cost = opt_cost_from_discrete_seq(maybe_new_text+mw+chr(new_unicode_point))
            if new_cost < best_cost:
                if maybe_new_text.count(chr(new_unicode_point))==0:
                    breakpoint()
                text = maybe_new_text
                best_cost = opt_cost_from_discrete_seq(text)
                codebook[chr(new_unicode_point)] = mw, text.count(chr(new_unicode_point))
                available_unicode_points.remove(new_unicode_point)
    cbcounts = np.array([text.count(c) for c in codebook.keys()])
    codebook_cost = sum(opt_cost_from_discrete_seq(v[0]) for v in codebook.values())
    idxs_cost = opt_cost_from_counts(cbcounts)
    rand_masked_text = ''.join(c if c in codebook else 'x' for c in text)
    final_chars, final_counts = np.unique(list(rand_masked_text), return_counts=True)
    idxs_cost = opt_cost_from_counts(final_counts)
    residual_str = ''.join(c for c in text if c not in codebook.keys())
    residual_cost = opt_cost_from_discrete_seq(residual_str)
    total_cost = codebook_cost + idxs_cost + residual_cost
    lccscore = codebook_cost + idxs_cost

    results = {'Model Cost': codebook_cost, 'Idx Cost': idxs_cost, 'Residual Cost': residual_cost, 'Total': total_cost, 'LCCScore': lccscore}
    return results

