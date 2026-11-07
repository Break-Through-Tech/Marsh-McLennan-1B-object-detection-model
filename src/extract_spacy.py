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
	"worker", "employee", "coworker", "cyclist", "pedestrian", "passenger",
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

# Modifiers that identify a type or an injury-relevant condition. Ordinary
# size, color and appearance adjectives are usually unnecessary.
IDENTIFYING_MODIFIERS = {
	"boiling", "carpeted", "chemical", "electric", "electrical", "frozen",
	"industrial", "medical", "motorized", "plastic", "prosthetic", "rubber",
	"slippery", "wet", "wooden",
}
THERMAL_MODIFIERS = {"cold", "hot"}
THERMAL_HEADS = {"coffee", "grease", "oil", "steam", "tea", "tub", "water", "heater"}
PRODUCT_MODIFIERS = {"baby", "child", "infant"}
PLACE_HEADS = {"bathroom", "bedroom", "garage", "kitchen", "room"}
CLINICAL_GEAR = {"bandage", "boot", "brace", "cast", "crutch", "dressing", "splint", "wrap"}
MEDICAL_TRANSPORT = {"ambulance", "gurney", "stretcher"}
TRANSPORT_ACTIONS = {"bring", "carry", "drive", "fly", "take", "transport"}
NONPHYSICAL_HEADS = {"accident", "incident", "injury", "present"}
AMBIGUOUS_ACTIVITIES = {"baseball", "basketball", "football", "hockey", "soccer", "softball", "volleyball"}


def is_verb_modifier(token) -> bool:
	"""True for -ing verb forms glued onto an object, as in "playing football"."""
	if token.text in KEEP_MODIFIERS:
		return False
	# The tagger labels most of these NN, so the VBG tag alone misses them.
	return token.tag_ == "VBG" or token.text.endswith("ing")


def is_absent(root) -> bool:
	"""Check the noun's local scope for explicit absence."""
	token = root
	while True:
		if any(child.lower_ == "no" and child.dep_ in {"det", "neg"} for child in token.children):
			return True
		if token.dep_ != "conj" or token.head.pos_ not in {"NOUN", "PROPN"}:
			break
		if any(child.lower_ in {"but", "yet"} for noun in (token, token.head)
				for child in noun.children if token.head.i < child.i < token.i):
			return False
		token = token.head
	if token.dep_ == "pobj":
		return token.head.lower_ == "without"
	if token.dep_ in {"dobj", "obj"}:
		verb = token.head
		return verb.lemma_ == "deny" or (
			verb.lemma_ in {"carry", "have", "use", "wear"}
			and any(child.dep_ == "neg" for child in verb.children)
		)
	return False


def is_post_injury_context(root, chunk) -> bool:
	"""Recognize conveyance and clearly marked clinical interventions."""
	preposition = root.head
	governor = preposition.head if root.dep_ == "pobj" else preposition
	if (
		root.dep_ == "pobj"
		and preposition.lower_ in {"by", "via", "in", "on"}
		and root.lemma_ in MEDICAL_TRANSPORT
		and governor.lemma_ in TRANSPORT_ACTIONS
	):
		return True
	if root.lemma_ not in CLINICAL_GEAR:
		return False
	if any(child.dep_ == "mark" and child.lower_ in {"while", "before"} for child in governor.children):
		return False
	if governor.lemma_ in {"apply", "administer", "prescribe", "provide", "receive"}:
		return any(token.lower_ in {"medical", "orthopedic", "surgical"} for token in chunk) or (
			root.lemma_ in {"bandage", "brace", "cast", "crutch", "dressing", "splint"}
		)
	return preposition.lower_ == "in" and governor.lemma_ in {"discharge", "immobilize", "treat"}


def object_phrase(root, chunk) -> str:
	"""Keep nested compounds and identifying modifiers attached to the candidate."""
	modifiers = []
	for token in root.children:
		if not chunk.start <= token.i < root.i or not token.is_alpha:
			continue
		if token.dep_ == "compound" and (
			token.lemma_ in PRODUCT_MODIFIERS
			or (token.lemma_ not in STOP_LEMMAS and not is_verb_modifier(token))
		):
			modifiers.append((token.i, object_phrase(token, chunk)))
		elif token.dep_ in {"amod", "compound", "acl"} and (
			token.lower_ in IDENTIFYING_MODIFIERS
			or (token.lower_ in THERMAL_MODIFIERS and root.lemma_ in THERMAL_HEADS)
		):
			modifiers.append((token.i, token.lower_))
	return " ".join([word for _, word in sorted(modifiers)] + [root.lemma_])


def extract_objects(doc) -> list:
	"""Return object phrases from one parsed narrative, in order of mention."""
	objects = []
	for chunk in doc.noun_chunks:
		root = chunk.root
		if root.pos_ != "NOUN" or not root.is_alpha or len(root.text) < 3:
			continue
		if is_absent(root) or is_post_injury_context(root, chunk):
			continue
		if root.lemma_ in PLACE_HEADS:
			continue
		candidates = [root]
		if root.lemma_ in STOP_LEMMAS:
			if root.lemma_ in NONPHYSICAL_HEADS:
				candidates = [
					child for child in root.lefts
					if child.i >= chunk.start and child.dep_ == "compound"
					and child.pos_ == "NOUN" and child.is_alpha
					and child.lemma_ not in STOP_LEMMAS
					and child.lemma_ not in AMBIGUOUS_ACTIVITIES and not is_verb_modifier(child)
				]
			elif not (
				(root.lemma_ == "weight" and root.head.lemma_ in {"drop", "fall", "lift", "throw"}
					and not any(child.lower_ in {"body", "own", "total"} for child in root.lefts))
				or (root.lemma_ in BODY_PARTS and any(child.lower_ == "prosthetic" for child in root.lefts))
			):
				continue
		for candidate in candidates:
			phrase = object_phrase(candidate, chunk)
			if candidate.text.endswith("ing") and " " in phrase and is_verb_modifier(candidate):
				phrase = phrase.rsplit(" ", 1)[0]
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
