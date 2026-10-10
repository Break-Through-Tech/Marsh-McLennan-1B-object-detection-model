import argparse
from pathlib import Path

import pandas as pd
import spacy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_INPUT = DATA_DIR / "gold" / "gold_sample.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "predictions_spacy.csv"

# Nouns that are never objects under docs/object-definition.md. These lists are
# the main thing to tune when reviewing errors against the gold set.
PEOPLE = {
	"pt", "patient", "mom", "mother", "dad", "father", "parent", "brother",
	"sister", "sibling", "son", "daughter", "child", "kid", "baby", "infant",
	"friend", "husband", "wife", "spouse", "family", "caregiver", "staff",
	"nurse", "person", "people", "man", "woman", "boy", "girl", "cousin",
	"grandmother", "grandfather", "grandma", "grandpa", "neighbor", "player",
	"teammate", "classmate", "student", "teacher", "coach", "male", "female",
}
BODY_PARTS = {
	"head", "face", "forehead", "scalp", "eye", "eyebrow", "ear", "nose",
	"mouth", "lip", "tongue", "tooth", "chin", "jaw", "neck", "shoulder",
	"arm", "forearm", "elbow", "wrist", "hand", "finger", "thumb", "chest",
	"rib", "back", "abdomen", "stomach", "hip", "pelvis", "groin", "buttock",
	"leg", "thigh", "knee", "shin", "ankle", "foot", "toe", "heel", "skin",
	"body", "extremity", "side", "throat", "nail", "fingernail", "toenail",
}
NOT_OBJECTS = {
	# injuries, symptoms and clinical shorthand
	"pain", "injury", "fall", "laceration", "lac", "fracture", "fx", "sprain",
	"strain", "contusion", "abrasion", "swelling", "bruise", "bruising",
	"bleeding", "wound", "burn", "cut", "bump", "loc", "chi", "syncope",
	"dizziness", "weakness", "hematoma", "concussion", "loss", "consciousness",
	"c", "o", "p", "s", "w", "h", "hx", "history", "scan", "xray", "x", "ed",
	"er", "hospital", "ems", "complaint", "symptom", "blood", "thinner",
	"foreign", "fb", "episode", "onset", "sensation", "deformity", "trauma",
	# time, place and other generic nouns
	"day", "night", "morning", "evening", "afternoon", "today", "yesterday",
	"week", "month", "year", "hour", "minute", "time", "ago", "prior",
	"home", "house", "school", "work", "park", "facility", "nursing",
	"way", "thing", "something", "anything", "top", "bottom", "edge",
	"front", "area", "part", "end", "piece", "accident", "incident", "event",
	"game", "practice", "activity", "arrival", "weight", "height", "ft",
	"inch", "lot", "pair", "couple", "type", "kind", "present", "state",
	"pta", "none", "seizure", "breath", "shortness", "life", "driver",
	# junk found when reviewing predictions on the gold sample
	"complain", "report", "evaluation", "nausea", "palpation", "tear", "mark",
	"care", "strike", "return", "process", "improvement", "numbness",
	"concern", "contact", "information", "behavior", "distress",
	"conversation", "status", "outburst", "hit", "trip", "slip", "pop",
	"rotation", "cont", "inj", "sign", "downtime", "help", "call", "walk",
	"turn", "middle", "direction", "speed", "region", "self", "place", "angle",
	"discoloration", "redness", "inflammation", "baseline", "disability",
	"honeymoon", "heartrate", "warmth", "rash", "exam", "note",
}
STOP_LEMMAS = PEOPLE | BODY_PARTS | NOT_OBJECTS

# Clinical shorthand that the parser misreads; expand or drop before parsing.
SHORTHAND = {
	r"\bw/\s*": "with ",
	r"\bc/o\b": "complains of",
	r"\bs/p\b": "after",
	r"\bp/w\b": "presents with",
	r"\bpt\b": "patient",
}

# Modifiers that look like verb forms but name the object: "swimming pool".
KEEP_MODIFIERS = {"swimming", "waiting", "boiling", "cutting", "ironing", "hot"}


def is_verb_modifier(token) -> bool:
	"""True for -ing verb forms glued onto an object, as in "playing football"."""
	if token.text in KEEP_MODIFIERS:
		return False
	# The tagger labels most of these NN, so the VBG tag alone misses them.
	return token.tag_ == "VBG" or token.text.endswith("ing")


def extract_objects(doc) -> list:
	"""Return object phrases from one parsed narrative, in order of mention."""
	objects = []
	for chunk in doc.noun_chunks:
		root = chunk.root
		if root.pos_ != "NOUN" or not root.is_alpha or len(root.text) < 3:
			continue
		if root.lemma_ in STOP_LEMMAS:
			continue
		# Keep compound modifiers so "car seat" is not reduced to "seat".
		words = [
			token.lemma_
			for token in chunk
			if token.dep_ == "compound"
			and token.head == root
			and token.is_alpha
			and token.lemma_ not in STOP_LEMMAS
			and not is_verb_modifier(token)
		]
		# The parser tags -ing verb forms as nouns and glues them onto the
		# object ("dresser complaining"); keep only the object words.
		if words and root.text.endswith("ing"):
			phrase = " ".join(words)
		else:
			phrase = " ".join(words + [root.lemma_])
		if phrase not in objects:
			objects.append(phrase)
	return objects


def predict(texts: pd.Series) -> list:
	nlp = spacy.load("en_core_web_sm", disable=["ner"])
	# Narratives are all caps, which the tagger handles badly; lowercase first.
	texts = texts.fillna("").str.lower()
	for pattern, replacement in SHORTHAND.items():
		texts = texts.str.replace(pattern, replacement, regex=True)
	docs = nlp.pipe(texts, batch_size=256)
	return ["; ".join(extract_objects(doc)) for doc in docs]


def main() -> None:
	parser = argparse.ArgumentParser(description="spaCy noun-chunk object extractor.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--limit", type=int, help="Only process the first N rows.")
	args = parser.parse_args()

	data = pd.read_csv(args.input, nrows=args.limit)
	predictions = pd.DataFrame(
		{"case_id": data["case_id"], "objects": predict(data["incident_text"])}
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	predictions.to_csv(args.output, index=False)
	print(f"Saved predictions for {len(predictions)} narratives to {args.output.name}.")


if __name__ == "__main__":
	main()
