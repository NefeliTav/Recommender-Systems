from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.hf_model import GENRE
model = GENRE.from_pretrained(
    "./hf_e2e_entity_linking_wiki_abs").eval()

sentences = ["In 1921, Einstein received a Nobel Prize."]

predictions = model.sample(sentences)
print(predictions)
