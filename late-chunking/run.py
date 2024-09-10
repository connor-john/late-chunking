from transformers import AutoModel
from transformers import AutoTokenizer
import numpy as np


def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids(".")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    token_offsets = inputs["offset_mapping"][0]
    token_ids = inputs["input_ids"][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def chunked_pooling(model_output, span_annotation: list, max_length=None):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def main():
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )

    input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."

    # determine chunks
    chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
    print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')

    # chunk before
    embeddings_traditional_chunking = model.encode(chunks)

    # chunk afterwards (context-sensitive chunked pooling)
    inputs = tokenizer(input_text, return_tensors="pt")
    model_output = model(**inputs)
    embeddings = chunked_pooling(model_output, [span_annotations])[0]

    berlin_embedding = model.encode("Berlin")

    for chunk, new_embedding, trad_embeddings in zip(
        chunks, embeddings, embeddings_traditional_chunking
    ):
        print(
            f'similarity_new("Berlin", "{chunk}"):',
            cosine_similarity(berlin_embedding, new_embedding),
        )
        print(
            f'similarity_trad("Berlin", "{chunk}"):',
            cosine_similarity(berlin_embedding, trad_embeddings),
        )


if __name__ == "__main__":
    main()
