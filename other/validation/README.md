# Validating Results

PSA: This is old code and not meant to be super maintained. More general guideline.

This is an extremely helpful validation tool for checking the results of your WebGPU model versus the original when writing kernels or otherwise.

The format is an array of model states at each point in a generation sequence, first generating from the reference model and saving the state of the model as you generate each token, then comparing to the browser model. You must greedily select tokens, of course, to maintain determinism. This can be done simply by setting top_k = 1.

I haven't included a script for how to export this generation as my code was quite sloppy and this will likely be quite different depending on your implementation. Here's an example of how you might save from Andrej Karpathy's NanoGPT code:

```

def generate(self, idx, max_new_tokens, temperature=1.0, top_k=1):
    for i in range(max_new_tokens):

        # I sloppily made a global that tracks the generation index.
        index = i

        # Another global variable.
        tensors.append({})

        idx_cond = idx if idx.size(
            1) <= self.config.block_size else idx[:, -self.config.block_size:]

        # Save inputs.
        logits, _ = self(idx_cond)

        # Save the logits.
        tensors[index]['logits'] = logits

        logits = logits[:, -1, :] / temperature
        tensors[index]['logits_t'] = logits

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        # Save the probs.
        tensors[index]['probs'] = probs

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    # Save tensors to JSON + format them correctly.
    # See conversion scripts for correct formatting.
```

# Included validation files.

I've included 2 validation files (gpt2medium_validation.json and shakespeare_validation.json) for convenience.

Both are sampled with the prompt "What is the answer to life, the universe, and everything?".
