import argparse
import random
import re
import statistics
from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch

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
	"coccyx", "tailbone",
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
	"industrial", "lanyard", "medical", "motorized", "plastic", "prosthetic", "rubber",
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


# Keep unresolved labeling choices together. These defaults include physical
# animals but exclude activity names and equipment merely worn in the scene.
INCLUDE_ANIMALS = True
INCLUDE_WORN_EQUIPMENT = False
ANIMALS = {"dog", "cat", "horse", "bird", "snake", "insect"}
CONTACT_ACTIONS = {
	"hit", "strike", "punch", "kick", "cut", "burn", "bite", "scratch",
	"drop", "spill", "swallow", "chew", "catch", "jam", "crush", "pull",
}
USE_ACTIONS = {"use", "ride", "climb", "lift", "grab", "hold", "operate", "walk", "sit", "stand"}
RELATION_PREPOSITIONS = {
	"fall": {"from", "off", "out", "on", "onto", "down", "over", "against", "into", "to"},
	"land": {"on", "onto", "against"},
	"trip": {"on", "over"}, "slip": {"on", "in", "off"},
	"step": {"on", "onto", "in"}, "collide": {"with"},
	"crash": {"into", "against", "on"}, "get": {"out", "off"},
}
SPATIAL_PARTS = {"corner", "edge", "top", "bottom", "side", "end"}
CONTACT_PREPOSITIONS = {"on", "onto", "against", "into", "with", "by", "in", "between"}
SURFACE_STRUCTURE_HEADS = {
	"floor", "wall", "stair", "stairs", "step", "ground", "road", "railing", "surface",
}
INJURY_EVENT_ACTIONS = CONTACT_ACTIONS | {
	"collapse", "fall", "land", "slip", "trip",
}
SURFACE_EVENT_TERMS = INJURY_EVENT_ACTIONS | {"foosh"}
SURFACE_RELATION_PREPOSITIONS = {
	"against", "at", "down", "from", "in", "into", "off", "on", "onto", "to",
}
TEMPORAL_MARKERS = {"as", "before", "when", "while"}
POSITION_PREPOSITIONS = {"against", "in", "on"}
ACTION_LEMMA_OVERRIDES = {"fell": "fall"}
MISPARSED_NOUN_HEADS = {"can"}


def action_lemma(token) -> str:
	"""Normalize common action forms even when the parser assigns the wrong POS."""
	return ACTION_LEMMA_OVERRIDES.get(token.lower_, token.lemma_)


def has_temporal_injury_event(governor) -> bool:
	"""Check for an explicitly time-linked injury event under a state verb."""
	for clause in governor.children:
		if clause.dep_ not in {"advcl", "ccomp", "conj"}:
			continue
		branch = tuple(clause.subtree)
		if not any(token.lower_ in TEMPORAL_MARKERS for token in branch):
			continue
		if any(
			action_lemma(token) in INJURY_EVENT_ACTIONS
			and (token.pos_ in {"VERB", "AUX"} or token.lower_ in ACTION_LEMMA_OVERRIDES)
			for token in branch
		):
			return True
	return False


def is_relevant(root) -> bool:
	"""Require a local dependency relationship, not just a nearby injury word.

	Follow noun coordination, body-part attachment (hit head on wall), and
	nested prepositions (fell out of bed), but never cross another verb.
	"""
	if not INCLUDE_ANIMALS and root.lemma_ in ANIMALS:
		return False
	# A misattached participle still provides direct local evidence ("spilled
	# hot water"); do not search arbitrary words elsewhere in the sentence.
	if any(child.lemma_ in CONTACT_ACTIONS and child.dep_ in {"amod", "acl"}
			and child.pos_ == "VERB" and not any(c.dep_ == "neg" for c in child.children)
			for child in root.children):
		return True
	token = root
	while token.dep_ == "conj" and token.head.pos_ in {"NOUN", "PROPN"}:
		token = token.head
	prepositions = []
	for _ in range(6):
		head = token.head
		if head == token:
			return False
		if head.pos_ in {"VERB", "AUX"}:
			action = action_lemma(head)
			if any(child.dep_ == "neg" for child in head.children):
				return False
			if not prepositions:
				if token.dep_ in {"nsubj", "nsubjpass"}:
					return action in CONTACT_ACTIONS | {"fall", "break", "collapse"}
				if (
					root.lemma_ in SURFACE_STRUCTURE_HEADS
					and token.dep_ == "npadvmod"
					and action in {"fall", "land", "slip", "trip"}
				):
					return True
				return token.dep_ in {"dobj", "obj"} and (
					action in CONTACT_ACTIONS | USE_ACTIONS
					or (action == "wear" and INCLUDE_WORN_EQUIPMENT)
				)
			prep = prepositions[-1]
			return (
				(action in CONTACT_ACTIONS and prep in CONTACT_PREPOSITIONS)
				or prep in RELATION_PREPOSITIONS.get(action, set())
				or (action in USE_ACTIONS and prep in {"on", "with", "up", "down"})
				or (
					action == "be"
					and prep in POSITION_PREPOSITIONS
					and has_temporal_injury_event(head)
				)
			)
		if head.pos_ == "ADP":
			prepositions.append(head.lower_)
		elif (
			root.lemma_ in SURFACE_STRUCTURE_HEADS
			and prepositions
			and action_lemma(head) in SURFACE_EVENT_TERMS
		):
			return prepositions[-1] in SURFACE_RELATION_PREPOSITIONS
		elif (
			root.lemma_ in SURFACE_STRUCTURE_HEADS
			and head.lower_ in IDENTIFYING_MODIFIERS
			and head.pos_ in {"ADJ", "NOUN"}
		):
			pass
		elif head.lemma_ not in BODY_PARTS | SPATIAL_PARTS:
			return False
		token = head
	return False


# This optional downstream mapping never participates in extraction scoring,
# including when the experimental hybrid recovery layer is enabled.
CATEGORY_HEADS = {
	"SURFACE_STRUCTURE": SURFACE_STRUCTURE_HEADS,
	"FURNITURE_FIXTURE": {"bed", "chair", "table", "door", "frame", "couch", "heater"},
	"TOOL_EQUIPMENT": {"knife", "drill", "hammer", "ladder", "clip", "scissor"},
	"VEHICLE": {"bicycle", "bike", "scooter", "car", "truck", "motorcycle", "bus"},
	"SPORT_RECREATION": {"ball", "bat", "racket", "skateboard", "trampoline"},
	"SUBSTANCE": {"water", "oil", "grease", "coffee", "tea", "steam", "chemical"},
	"ANIMAL": ANIMALS,
}


def classify_objects(objects: list) -> dict:
	"""Assign provisional coarse categories after extraction; no accuracy claim."""
	return {
		phrase: next((category for category, heads in CATEGORY_HEADS.items()
			if phrase.split()[-1] in heads), "OTHER_PHYSICAL_OBJECT")
		for phrase in objects
	}


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
	words = [word for _, word in sorted(modifiers)] + [root.lemma_]
	if root.i + 1 < len(root.doc):
		follower = root.doc[root.i + 1]
		if (
			root.dep_ in {"dobj", "obj", "pobj"}
			and action_lemma(root.head) in CONTACT_ACTIONS | USE_ACTIONS
			and follower.lower_ in MISPARSED_NOUN_HEADS
			and follower.pos_ in {"AUX", "VERB"}
			and not tuple(follower.children)
			and (follower.head == root.head or follower.dep_ == "dep")
		):
			words.append(follower.lower_)
	return " ".join(words)


def is_generic_part_of_larger_object(root) -> bool:
	"""Suppress a generic part when an explicit larger physical whole follows."""
	if root.lemma_ not in SPATIAL_PARTS:
		return False
	for preposition in root.children:
		if preposition.dep_ != "prep" or preposition.lower_ != "of":
			continue
		if any(
			child.dep_ == "pobj"
			and child.pos_ in {"NOUN", "PROPN"}
			and child.lemma_ not in STOP_LEMMAS
			for child in preposition.children
		):
			return True
	return False


def extract_candidates(doc):
	"""Generate physical noun candidates without deciding injury relevance."""
	yielded = set()
	for chunk in doc.noun_chunks:
		root = chunk.root
		if root.pos_ != "NOUN" or not root.is_alpha or len(root.text) < 3:
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
			yielded.add(candidate.i)
			yield candidate, chunk

	# The parser can omit numeric path nouns ("fell down 6 steps") from noun
	# chunks or attach a surface under a mis-tagged condition ("floor slippery").
	# Recover only those locally checkable candidates; relevance is still
	# required below.
	for token in doc:
		if token.i in yielded or token.pos_ != "NOUN" or not token.is_alpha or len(token.text) < 3:
			continue
		if token.lemma_ in STOP_LEMMAS | PLACE_HEADS:
			continue
		if token.dep_ == "npadvmod" or token.lemma_ in SURFACE_STRUCTURE_HEADS:
			yield token, doc[token.i:token.i + 1]


def extract_objects(doc) -> list:
	"""Filter spaCy candidates, then normalize; preserve the prediction API."""
	objects = []
	for candidate, chunk in extract_candidates(doc):
		if (
			is_absent(candidate)
			or is_post_injury_context(candidate, chunk)
			or is_generic_part_of_larger_object(candidate)
		):
			continue
		if is_verb_modifier(candidate) or candidate.lemma_ in AMBIGUOUS_ACTIVITIES or not is_relevant(candidate):
			continue
		phrase = object_phrase(candidate, chunk)
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


# ============================================================
# HYBRID SPAN RECOVERY
# ============================================================
# The rule-based extractor has high precision but misses many
# true objects. This optional layer uses a trainable span model
# to recover likely injury-related object mentions that were
# missed by the strict relevance rules.
#
# It is intentionally a recovery layer rather than a replacement
# for the deterministic extractor.
# ============================================================

HYBRID_SPAN_LABEL = "INJURY_OBJECT"
HYBRID_SPANS_KEY = "injury_objects"
# Selected from near-best development micro-F1 values using held-out-fold
# precision as the tie-breaker; see HYBRID_SELECTION_F1_TOLERANCE below.
HYBRID_SPAN_THRESHOLD = 0.80
HYBRID_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80)
HYBRID_SELECTION_F1_TOLERANCE = 0.005
HYBRID_CV_FOLDS = 5
# Fixed development seed makes fold assignment and model initialization repeatable.
HYBRID_RANDOM_SEED = 20261108
HYBRID_FIXED_EPOCHS = 20
HYBRID_FIXED_TRAINING_MODE = "fixed"
HYBRID_EARLY_STOPPING_MODE = "early_stopping"
HYBRID_DEFAULT_TRAINING_MODE = HYBRID_FIXED_TRAINING_MODE
HYBRID_FINAL_RETRAIN_SEED_OFFSET = 10000
HYBRID_MAX_EPOCHS = 40
HYBRID_EARLY_STOPPING_PATIENCE = 5
HYBRID_MIN_DELTA = 0.001
HYBRID_VALIDATION_FRACTION = 0.20
# Epoch selection needs one fixed threshold; the final threshold remains an
# independent choice made from aggregated outer-fold predictions.
HYBRID_INNER_VALIDATION_THRESHOLD = 0.50
HYBRID_MIN_ALIGNMENT_COVERAGE = 0.90


def _gold_phrases(value) -> list:
	"""Split an annotation cell without manufacturing missing object phrases."""
	if value is None or pd.isna(value):
		return []
	return [
		phrase for phrase in (" ".join(item.strip().split()) for item in str(value).split(";"))
		if phrase and phrase.upper() != "NONE"
	]


def _object_key(value) -> str:
	"""Mirror the evaluator's conservative whitespace and final-s normalization."""
	value = " ".join(str(value).lower().split())
	if value.endswith("s") and not value.endswith("ss"):
		value = value[:-1]
	return value


def _strict_char_matches(doc, pattern: str) -> list:
	"""Return only matches that coincide with tokenizer boundaries."""
	spans = []
	for match in re.finditer(pattern, doc.text, flags=re.IGNORECASE):
		span = doc.char_span(match.start(), match.end(), alignment_mode="strict")
		if span is not None:
			spans.append(span)
	return spans


def _safe_plural_matches(doc, phrase: str) -> list:
	"""Find token windows differing only by the evaluator's final-s rule."""
	tokens = list(re.finditer(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", doc.text))
	word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", phrase))
	if not word_count:
		return []
	matches = []
	for start in range(len(tokens) - word_count + 1):
		window = tokens[start:start + word_count]
		candidate = doc.text[window[0].start():window[-1].end()]
		if _object_key(candidate) != _object_key(phrase):
			continue
		span = doc.char_span(window[0].start(), window[-1].end(), alignment_mode="strict")
		if span is not None:
			matches.append(span)
	return matches


def align_gold_spans(text: str, gold_objects, tokenizer=None) -> dict:
	"""Conservatively align human phrases to literal narrative offsets."""
	if tokenizer is None:
		tokenizer = spacy.blank("en")
	doc = tokenizer.make_doc(str(text or ""))
	alignments = []
	unaligned = []
	seen_offsets = set()
	for phrase in _gold_phrases(gold_objects):
		methods = (
			("exact", re.escape(phrase)),
			("whitespace", r"\s+".join(re.escape(part) for part in phrase.split())),
		)
		matched_spans = []
		method_used = None
		for method, inner_pattern in methods:
			pattern = rf"(?<!\w){inner_pattern}(?!\w)"
			matched_spans = _strict_char_matches(doc, pattern)
			if matched_spans:
				method_used = method
				break
		if not matched_spans:
			matched_spans = _safe_plural_matches(doc, phrase)
			if matched_spans:
				method_used = "safe_plural"
		if not matched_spans:
			unaligned.append(phrase)
			continue
		phrase_offsets = []
		for span in matched_spans:
			offset = (span.start_char, span.end_char)
			if offset not in seen_offsets:
				seen_offsets.add(offset)
				phrase_offsets.append(offset)
		alignments.append({"phrase": phrase, "method": method_used, "offsets": phrase_offsets})
	return {"alignments": alignments, "unaligned": unaligned}


def audit_gold_span_alignment(data: pd.DataFrame) -> dict:
	"""Summarize supervision coverage before any span model is trained."""
	tokenizer = spacy.blank("en")
	counts = {"exact": 0, "whitespace": 0, "safe_plural": 0}
	unaligned = []
	total = 0
	for row in data.itertuples(index=False):
		phrases = _gold_phrases(row.gold_objects)
		total += len(phrases)
		result = align_gold_spans(row.incident_text, row.gold_objects, tokenizer)
		for alignment in result["alignments"]:
			counts[alignment["method"]] += 1
		unaligned.extend(
			{"case_id": row.case_id, "phrase": phrase, "incident_text": row.incident_text}
			for phrase in result["unaligned"]
		)
	alignable = sum(counts.values())
	return {
		"total": total,
		"exact": counts["exact"],
		"whitespace": counts["whitespace"],
		"safe_plural": counts["safe_plural"],
		"alignable": alignable,
		"unalignable": len(unaligned),
		"coverage": alignable / total if total else 1.0,
		"unaligned_examples": unaligned,
	}


def build_span_examples(nlp, records) -> tuple:
	"""Build span supervision, excluding rows with noisy/unresolved offsets."""
	examples = []
	skipped_case_ids = []
	for row in records:
		text = str(row["incident_text"] or "")
		alignment = align_gold_spans(text, row["gold_objects"], nlp)
		# An unaligned positive must not become an accidental negative example.
		if alignment["unaligned"]:
			skipped_case_ids.append(row["case_id"])
			continue
		spans = [
			(start, end, HYBRID_SPAN_LABEL)
			for item in alignment["alignments"]
			for start, end in item["offsets"]
		]
		doc = nlp.make_doc(text)
		examples.append(Example.from_dict(doc, {"spans": {HYBRID_SPANS_KEY: spans}}))
	return examples, skipped_case_ids


# ============================================================
# EARLY STOPPING / BEST-EPOCH SELECTION
# ============================================================
# The outer CV fold is reserved for unbiased development
# evaluation. Epoch selection therefore uses a separate inner
# validation set.
#
# We keep the model state with the best validation micro-F1 and
# stop when performance has not improved for `patience` epochs.
# The restored state verifies epoch selection, but only the epoch
# count is transferred to a fresh model trained on every usable
# outer-training case before the outer fold is predicted.
# ============================================================


def _inner_case_split(data: pd.DataFrame, validation_fraction: float, seed: int) -> tuple:
	"""Split whole case IDs into inner train and validation partitions."""
	if not 0 < validation_fraction < 1:
		raise ValueError("Inner validation fraction must be between zero and one.")
	case_ids = list(dict.fromkeys(data["case_id"].tolist()))
	if len(case_ids) < 2:
		raise ValueError("At least two cases are required for an inner validation split.")
	random.Random(seed).shuffle(case_ids)
	validation_count = max(1, round(len(case_ids) * validation_fraction))
	validation_count = min(validation_count, len(case_ids) - 1)
	validation_ids = set(case_ids[:validation_count])
	validation_mask = data["case_id"].isin(validation_ids)
	inner_train = data[~validation_mask].copy()
	inner_validation = data[validation_mask].copy()
	if inner_train.empty or inner_validation.empty:
		raise ValueError("Inner case split produced an empty partition.")
	return inner_train, inner_validation


def _initialize_span_recovery_training(data: pd.DataFrame, seed: int):
	"""Create a fresh SpanCategorizer, aligned examples, and optimizer."""
	fix_random_seed(seed)
	nlp = spacy.blank("en")
	nlp.add_pipe(
		"spancat",
		config={
			"spans_key": HYBRID_SPANS_KEY,
			# Keep scores for every proposed n-gram so thresholding stays external.
			"threshold": 0.0,
			"suggester": {"@misc": "spacy.ngram_suggester.v1", "sizes": [1, 2, 3]},
		},
	)
	examples, skipped_case_ids = build_span_examples(nlp, data.to_dict("records"))
	if not examples:
		raise ValueError("No safely aligned span examples are available for training.")
	optimizer = nlp.initialize(lambda: examples)
	return nlp, optimizer, examples, skipped_case_ids


def train_span_recovery_model(
	data: pd.DataFrame,
	validation_data: pd.DataFrame = None,
	validation_rule_predictions=None,
	max_epochs: int = HYBRID_MAX_EPOCHS,
	patience: int = HYBRID_EARLY_STOPPING_PATIENCE,
	min_delta: float = HYBRID_MIN_DELTA,
	validation_fraction: float = HYBRID_VALIDATION_FRACTION,
	seed: int = HYBRID_RANDOM_SEED,
	safety_nlp=None,
):
	"""Train explicitly, restore the best inner-validation epoch in memory."""
	if max_epochs < 1 or patience < 1:
		raise ValueError("max_epochs and patience must both be positive.")
	if validation_data is None:
		data, validation_data = _inner_case_split(data, validation_fraction, seed)
	data = data.copy()
	validation_data = validation_data.copy()
	if data.empty or validation_data.empty:
		raise ValueError("Span training requires non-empty inner train and validation data.")
	if set(data["case_id"]) & set(validation_data["case_id"]):
		raise ValueError("Inner train and validation case IDs must not overlap.")

	nlp, optimizer, examples, skipped_case_ids = _initialize_span_recovery_training(data, seed)
	rng = random.Random(seed)

	if safety_nlp is None:
		safety_nlp = spacy.load("en_core_web_sm", disable=["ner"])
	validation_texts = [
		"" if pd.isna(text) else str(text) for text in validation_data["incident_text"]
	]
	safety_texts = [
		text.lower() if len(text.lower()) == len(text) else text for text in validation_texts
	]
	# Parse validation narratives once; reparsing them after every epoch is wasteful.
	validation_safety_docs = list(safety_nlp.pipe(safety_texts, batch_size=128))
	if validation_rule_predictions is None:
		validation_rule_predictions = predict(validation_data["incident_text"])
	if len(validation_rule_predictions) != len(validation_data):
		raise ValueError("Validation rule predictions must align with validation rows.")

	best_epoch = None
	best_validation_f1 = float("-inf")
	best_validation_precision = 0.0
	best_validation_recall = 0.0
	best_model_state = None
	epochs_without_improvement = 0
	history = []
	for epoch in range(1, max_epochs + 1):
		rng.shuffle(examples)
		losses = {}
		for batch in minibatch(examples, size=16):
			nlp.update(batch, sgd=optimizer, drop=0.20, losses=losses)

		validation_candidates = predict_recovery_spans(
			nlp,
			validation_texts,
			safety_docs=validation_safety_docs,
		)
		validation_predictions = combine_rule_and_span_predictions(
			validation_rule_predictions,
			validation_candidates,
			threshold=HYBRID_INNER_VALIDATION_THRESHOLD,
		)
		validation_metrics = _score_object_predictions(
			validation_data["gold_objects"], validation_predictions
		)
		validation_f1 = validation_metrics["micro_f1"]
		history.append(
			{
				"epoch": epoch,
				"training_loss": float(losses.get("spancat", 0.0)),
				"validation_precision": validation_metrics["precision"],
				"validation_recall": validation_metrics["recall"],
				"validation_f1": validation_f1,
			}
		)
		if best_model_state is None or validation_f1 > best_validation_f1 + min_delta:
			best_epoch = epoch
			best_validation_f1 = validation_f1
			best_validation_precision = validation_metrics["precision"]
			best_validation_recall = validation_metrics["recall"]
			best_model_state = nlp.to_bytes()
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				break

	# Epoch one always initializes this state, even when validation F1 is zero.
	if best_model_state is None or best_epoch is None:
		raise RuntimeError("No best SpanCategorizer state was captured.")
	nlp.from_bytes(best_model_state)
	restored_candidates = predict_recovery_spans(
		nlp,
		validation_texts,
		safety_docs=validation_safety_docs,
	)
	restored_predictions = combine_rule_and_span_predictions(
		validation_rule_predictions,
		restored_candidates,
		threshold=HYBRID_INNER_VALIDATION_THRESHOLD,
	)
	restored_validation_f1 = _score_object_predictions(
		validation_data["gold_objects"], restored_predictions
	)["micro_f1"]
	if abs(restored_validation_f1 - best_validation_f1) > 1e-12:
		raise RuntimeError("Restored SpanCategorizer state does not match the best epoch.")
	return nlp, {
		"best_epoch": best_epoch,
		"stopped_epoch": history[-1]["epoch"],
		"epochs_executed": len(history),
		"best_validation_f1": best_validation_f1,
		"best_validation_precision": best_validation_precision,
		"best_validation_recall": best_validation_recall,
		"restored_validation_f1": restored_validation_f1,
		"best_model_state_restored": True,
		"stopped_early": len(history) < max_epochs,
		"skipped_case_ids": skipped_case_ids,
		"inner_train_cases": data["case_id"].nunique(),
		"inner_validation_cases": validation_data["case_id"].nunique(),
		"history": history,
	}


def train_span_recovery_full_data(data: pd.DataFrame, epochs: int, seed: int):
	"""Train a fresh fold model on all usable outer-training cases."""
	if epochs < 1:
		raise ValueError("Full-data retraining requires at least one epoch.")
	nlp, optimizer, examples, skipped_case_ids = _initialize_span_recovery_training(
		data.copy(), seed
	)
	rng = random.Random(seed)
	history = []
	for epoch in range(1, epochs + 1):
		rng.shuffle(examples)
		losses = {}
		for batch in minibatch(examples, size=16):
			nlp.update(batch, sgd=optimizer, drop=0.20, losses=losses)
		history.append(
		{"epoch": epoch, "training_loss": float(losses.get("spancat", 0.0))}
		)
	return nlp, {
		"epochs": epochs,
		"training_cases": data["case_id"].nunique(),
		"usable_training_cases": len(examples),
		"skipped_case_ids": skipped_case_ids,
		"history": history,
	}


def _normalize_recovery_candidate(safety_doc, start_char: int, end_char: int):
	"""Apply broad safety exclusions without repeating strict relevance rules."""
	span = safety_doc.char_span(start_char, end_char, alignment_mode="strict")
	if span is None:
		return None
	tokens = list(span)
	while tokens and (tokens[0].is_punct or tokens[0].pos_ in {"DET", "PRON"}):
		tokens.pop(0)
	while tokens and tokens[-1].is_punct:
		tokens.pop()
	while len(tokens) > 1 and tokens[0].lemma_ in PEOPLE and tokens[0].lemma_ not in PRODUCT_MODIFIERS:
		tokens.pop(0)
	if not tokens or not any(token.is_alpha for token in tokens):
		return None
	trimmed = safety_doc[tokens[0].i:tokens[-1].i + 1]
	head = tokens[-1]
	head_lemma = head.lemma_.lower() if head.lemma_ else head.lower_
	prosthetic = any(token.lower_ == "prosthetic" for token in tokens)
	if (
		head_lemma in PEOPLE | NOT_OBJECTS | PLACE_HEADS | AMBIGUOUS_ACTIVITIES
		or (head_lemma in BODY_PARTS and not prosthetic)
		or (not INCLUDE_ANIMALS and head_lemma in ANIMALS)
		or is_absent(head)
		or is_post_injury_context(head, trimmed)
		or is_generic_part_of_larger_object(head)
	):
		return None
	phrase = " ".join(token.lower_ for token in tokens).strip()
	return phrase or None


def predict_recovery_spans(model, texts, safety_nlp=None, safety_docs=None) -> list:
	"""Return safety-filtered recovery candidates with model confidence scores."""
	texts = ["" if pd.isna(text) else str(text) for text in texts]
	model_docs = model.pipe(texts, batch_size=128)
	if safety_docs is None:
		if safety_nlp is None:
			safety_nlp = spacy.load("en_core_web_sm", disable=["ner"])
		# Lowercasing helps the safety parser while preserving ASCII character offsets.
		safety_texts = [text.lower() if len(text.lower()) == len(text) else text for text in texts]
		safety_docs = safety_nlp.pipe(safety_texts, batch_size=128)
	else:
		if len(safety_docs) != len(texts):
			raise ValueError("Preparsed safety documents must align with input texts.")
	all_candidates = []
	for model_doc, safety_doc in zip(model_docs, safety_docs):
		spans = model_doc.spans.get(HYBRID_SPANS_KEY, [])
		scores = spans.attrs.get("scores", [])
		best_by_phrase = {}
		for span, score in zip(spans, scores):
			phrase = _normalize_recovery_candidate(
				safety_doc, span.start_char, span.end_char
			)
			if phrase:
				best_by_phrase[phrase] = max(best_by_phrase.get(phrase, 0.0), float(score))
		all_candidates.append(sorted(best_by_phrase.items(), key=lambda item: (-item[1], item[0])))
	return all_candidates


def combine_rule_and_span_predictions(
	rule_predictions,
	recovery_predictions,
	threshold: float = HYBRID_SPAN_THRESHOLD,
) -> list:
	"""Union high-precision rules with thresholded, safety-filtered recovery spans."""
	if len(rule_predictions) != len(recovery_predictions):
		raise ValueError("Rule and recovery prediction counts must match.")
	combined = []
	for rule_cell, recovered in zip(rule_predictions, recovery_predictions):
		objects = [item.strip() for item in str(rule_cell or "").split(";") if item.strip()]
		seen = {_object_key(item) for item in objects}
		for phrase, score in recovered:
			key = _object_key(phrase)
			if score >= threshold and key and key not in seen:
				objects.append(phrase)
				seen.add(key)
		combined.append("; ".join(objects))
	return combined


def _score_object_predictions(gold_cells, prediction_cells) -> dict:
	"""Score with the same set semantics used by the existing evaluator."""
	tp = fp = fn = exact = at_least_one = 0
	for gold_cell, prediction_cell in zip(gold_cells, prediction_cells):
		gold = {_object_key(item) for item in _gold_phrases(gold_cell)}
		predicted = {
			_object_key(item) for item in str(prediction_cell or "").split(";") if item.strip()
		}
		tp += len(gold & predicted)
		fp += len(predicted - gold)
		fn += len(gold - predicted)
		exact += gold == predicted
		at_least_one += bool(gold & predicted) if gold else not predicted
	precision = tp / (tp + fp) if tp + fp else 0.0
	recall = tp / (tp + fn) if tp + fn else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	n = len(gold_cells)
	return {
		"precision": precision,
		"recall": recall,
		"micro_f1": f1,
		"exact_match": exact / n if n else 0.0,
		"at_least_one": at_least_one / n if n else 0.0,
		"tp": tp,
		"fp": fp,
		"fn": fn,
	}


def _case_folds(data: pd.DataFrame, n_folds: int, seed: int) -> list:
	"""Assign each unique case to exactly one reproducible held-out fold."""
	if data["case_id"].isna().any() or data["case_id"].duplicated().any():
		raise ValueError("Cross-validation requires unique, non-empty case_id values.")
	indices = list(range(len(data)))
	random.Random(seed).shuffle(indices)
	return [indices[offset::n_folds] for offset in range(n_folds)]


def _inner_stability_summary(fold_training: list, patience: int) -> dict:
	best_epochs = [info["best_epoch"] for info in fold_training]
	validation_f1s = [info["best_validation_f1"] for info in fold_training]
	stopped_epochs = [info["stopped_epoch"] for info in fold_training]
	return {
		"best_epoch_mean": statistics.mean(best_epochs),
		"best_epoch_median": statistics.median(best_epochs),
		"best_epoch_min": min(best_epochs),
		"best_epoch_max": max(best_epochs),
		"best_epoch_range": max(best_epochs) - min(best_epochs),
		"best_epoch_std": statistics.pstdev(best_epochs),
		"validation_f1_mean": statistics.mean(validation_f1s),
		"validation_f1_range": max(validation_f1s) - min(validation_f1s),
		"validation_f1_std": statistics.pstdev(validation_f1s),
		"epoch_one_folds": sum(epoch == 1 for epoch in best_epochs),
		# Stopping within two patience windows indicates a very short/noisy curve.
		"quick_patience_stops": sum(epoch <= 2 * patience for epoch in stopped_epochs),
		"selection_epochs_executed": sum(info["epochs_executed"] for info in fold_training),
	}


def compare_inner_validation_fractions(
	data: pd.DataFrame,
	fractions=(0.10, 0.20),
	n_folds: int = HYBRID_CV_FOLDS,
	seed: int = HYBRID_RANDOM_SEED,
	max_epochs: int = HYBRID_MAX_EPOCHS,
	patience: int = HYBRID_EARLY_STOPPING_PATIENCE,
	min_delta: float = HYBRID_MIN_DELTA,
) -> dict:
	"""Compare inner-selection stability without evaluating any outer fold."""
	data = data.reset_index(drop=True).copy()
	folds = _case_folds(data, n_folds, seed)
	rule_predictions = predict(data["incident_text"])
	safety_nlp = spacy.load("en_core_web_sm", disable=["ner"])
	all_indices = set(range(len(data)))
	results = {}
	for fraction in fractions:
		fold_training = []
		for fold_number, held_out_indices in enumerate(folds):
			training_indices = sorted(all_indices - set(held_out_indices))
			outer_training = data.iloc[training_indices]
			inner_train, inner_validation = _inner_case_split(
				outer_training, fraction, seed + fold_number
			)
			validation_rule_predictions = [
				rule_predictions[index] for index in inner_validation.index
			]
			_, training_info = train_span_recovery_model(
				inner_train,
				validation_data=inner_validation,
				validation_rule_predictions=validation_rule_predictions,
				max_epochs=max_epochs,
				patience=patience,
				min_delta=min_delta,
				validation_fraction=fraction,
				seed=seed + fold_number,
				safety_nlp=safety_nlp,
			)
			fold_training.append(
				{
					"fold": fold_number + 1,
					"validation_fraction": fraction,
					"outer_training_cases": outer_training["case_id"].nunique(),
					"outer_held_out_cases": len(held_out_indices),
					**training_info,
				}
			)
		results[fraction] = {
			"fold_training": fold_training,
			"stability": _inner_stability_summary(fold_training, patience),
		}
	return {"folds": folds, "fractions": results}


def cross_validate_span_recovery(
	data: pd.DataFrame,
	thresholds=HYBRID_THRESHOLD_GRID,
	n_folds: int = HYBRID_CV_FOLDS,
	seed: int = HYBRID_RANDOM_SEED,
	training_mode: str = HYBRID_DEFAULT_TRAINING_MODE,
	fixed_epochs: int = HYBRID_FIXED_EPOCHS,
	max_epochs: int = HYBRID_MAX_EPOCHS,
	patience: int = HYBRID_EARLY_STOPPING_PATIENCE,
	min_delta: float = HYBRID_MIN_DELTA,
	validation_fraction: float = HYBRID_VALIDATION_FRACTION,
	preselected_fold_training=None,
) -> dict:
	"""Produce case-held-out span predictions with an explicit training policy.

	Fixed 20-epoch training currently has the strongest legacy-development F1 and
	therefore remains the recommended experimental baseline. Nested early stopping
	is retained as an opt-in research alternative; a future unseen holdout must
	decide which behavior generalizes.
	"""
	required = {"case_id", "incident_text", "gold_objects"}
	if not required.issubset(data.columns):
		raise ValueError(f"Missing required columns: {sorted(required - set(data.columns))}")
	valid_modes = {HYBRID_FIXED_TRAINING_MODE, HYBRID_EARLY_STOPPING_MODE}
	if training_mode not in valid_modes:
		raise ValueError(f"training_mode must be one of {sorted(valid_modes)}")
	if fixed_epochs < 1:
		raise ValueError("fixed_epochs must be at least one.")
	if preselected_fold_training is not None and training_mode != HYBRID_EARLY_STOPPING_MODE:
		raise ValueError("Preselected epochs are only valid for early-stopping training.")
	data = data.reset_index(drop=True).copy()
	alignment = audit_gold_span_alignment(data)
	if alignment["coverage"] < HYBRID_MIN_ALIGNMENT_COVERAGE:
		raise ValueError(
			f"Gold span alignment coverage {alignment['coverage']:.1%} is below "
			f"the required {HYBRID_MIN_ALIGNMENT_COVERAGE:.1%}."
		)
	folds = _case_folds(data, n_folds, seed)
	if preselected_fold_training is not None and len(preselected_fold_training) != len(folds):
		raise ValueError("Preselected epoch records must match the number of outer folds.")
	rule_predictions = predict(data["incident_text"])
	recovery_predictions = [None] * len(data)
	skipped_training_cases = set()
	fold_training = []
	safety_nlp = spacy.load("en_core_web_sm", disable=["ner"])
	all_indices = set(range(len(data)))
	for fold_number, held_out_indices in enumerate(folds):
		training_indices = sorted(all_indices - set(held_out_indices))
		outer_training = data.iloc[training_indices]
		if training_mode == HYBRID_FIXED_TRAINING_MODE:
			# Preserve the historically measured 0.786-F1 setup: one fresh model
			# sees every outer-training case for the same fixed number of epochs.
			final_seed = seed + fold_number
			model, final_info = train_span_recovery_full_data(
				outer_training, epochs=fixed_epochs, seed=final_seed
			)
			training_info = {
				"training_mode": training_mode,
				"selection_seed": None,
				"best_epoch": fixed_epochs,
				"stopped_epoch": fixed_epochs,
				"epochs_executed": 0,
				"best_validation_f1": None,
				"skipped_case_ids": [],
			}
		else:
			# Early stopping is intentionally an experimental alternative. The
			# outer fold remains untouched while an inner split selects the epoch.
			if preselected_fold_training is None:
				inner_train, inner_validation = _inner_case_split(
					outer_training, validation_fraction, seed + fold_number
				)
				validation_rule_predictions = [
					rule_predictions[index] for index in inner_validation.index
				]
				_, training_info = train_span_recovery_model(
					inner_train,
					validation_data=inner_validation,
					validation_rule_predictions=validation_rule_predictions,
					max_epochs=max_epochs,
					patience=patience,
					min_delta=min_delta,
					validation_fraction=validation_fraction,
					seed=seed + fold_number,
					safety_nlp=safety_nlp,
				)
			else:
				training_info = dict(preselected_fold_training[fold_number])
				if training_info.get("fold") != fold_number + 1:
					raise ValueError("Preselected epoch record does not match its outer fold.")
			training_info["training_mode"] = training_mode
			training_info["selection_seed"] = seed + fold_number

			# Epoch selection is complete before this fresh model sees the full
			# outer-training partition. The outer-held-out cases stay evaluation-only.
			final_seed = seed + HYBRID_FINAL_RETRAIN_SEED_OFFSET + fold_number
			model, final_info = train_span_recovery_full_data(
				outer_training, epochs=training_info["best_epoch"], seed=final_seed
			)
		skipped_training_cases.update(training_info["skipped_case_ids"])
		skipped_training_cases.update(final_info["skipped_case_ids"])
		fold_predictions = predict_recovery_spans(
			model, data.iloc[held_out_indices]["incident_text"], safety_nlp=safety_nlp
		)
		for index, candidates in zip(held_out_indices, fold_predictions):
			recovery_predictions[index] = candidates
		fold_training.append(
			{
				"fold": fold_number + 1,
				"validation_fraction": validation_fraction,
				"outer_training_cases": outer_training["case_id"].nunique(),
				"outer_held_out_cases": len(held_out_indices),
				**training_info,
				"final_retraining_seed": final_seed,
				"full_retraining_epochs": final_info["epochs"],
				"full_retraining_cases": final_info["training_cases"],
				"full_retraining_usable_cases": final_info["usable_training_cases"],
				"outer_prediction_count": len(fold_predictions),
			}
		)
	if any(candidates is None for candidates in recovery_predictions):
		raise RuntimeError("Every case must receive exactly one out-of-fold prediction.")
	threshold_results = {}
	for threshold in thresholds:
		hybrid = combine_rule_and_span_predictions(
			rule_predictions, recovery_predictions, threshold=threshold
		)
		threshold_results[threshold] = {
			"metrics": _score_object_predictions(data["gold_objects"], hybrid),
			"predictions": hybrid,
		}
	# This is explicit development tuning; a future unseen holdout must not select it.
	# Favor precision among effectively tied F1 values because recovery should not
	# undo the deterministic path's main strength over the original baseline.
	best_f1 = max(item["metrics"]["micro_f1"] for item in threshold_results.values())
	competitive_thresholds = [
		threshold for threshold, item in threshold_results.items()
		if best_f1 - item["metrics"]["micro_f1"] <= HYBRID_SELECTION_F1_TOLERANCE
	]
	best_threshold = max(
		competitive_thresholds,
		key=lambda threshold: (
			threshold_results[threshold]["metrics"]["precision"],
			threshold_results[threshold]["metrics"]["micro_f1"],
			threshold,
		),
	)
	maximum_possible_epochs = len(folds) * (
		fixed_epochs if training_mode == HYBRID_FIXED_TRAINING_MODE else max_epochs
	)
	selection_epochs = sum(info["epochs_executed"] for info in fold_training)
	full_retraining_epochs = sum(info["full_retraining_epochs"] for info in fold_training)
	actual_epochs = selection_epochs + full_retraining_epochs
	return {
		"training_mode": training_mode,
		"alignment": alignment,
		"folds": [[data.iloc[index]["case_id"] for index in fold] for fold in folds],
		"rule_predictions": rule_predictions,
		"recovery_predictions": recovery_predictions,
		"threshold_results": threshold_results,
		"best_threshold": best_threshold,
		"hybrid_predictions": threshold_results[best_threshold]["predictions"],
		"hybrid_metrics": threshold_results[best_threshold]["metrics"],
		"skipped_training_case_ids": sorted(skipped_training_cases, key=str),
		"fold_training": fold_training,
		"training_efficiency": {
			"maximum_possible_epochs": maximum_possible_epochs,
			"selection_epochs_executed": selection_epochs,
			"full_retraining_epochs": full_retraining_epochs,
			"actual_epochs_executed": actual_epochs,
			"epochs_avoided": maximum_possible_epochs - actual_epochs,
			"reduction_fraction": (
				(maximum_possible_epochs - actual_epochs) / maximum_possible_epochs
				if maximum_possible_epochs else 0.0
			),
		},
	}


# ============================================================
# HYBRID CANDIDATE RESCUE
# ============================================================
# The strict rule extractor has high precision but can discard
# real injury objects. This optional classifier reviews a broader
# candidate pool and rescues candidates that appear genuinely
# involved in the injury.
#
# It is a recovery layer, not a replacement for the deterministic
# extractor.
# ============================================================

# Selected by grouped out-of-fold development F1 with the precision-first
# tie-break defined in _select_candidate_threshold.
CANDIDATE_RESCUE_THRESHOLD = 0.70
CANDIDATE_RESCUE_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
CANDIDATE_RESCUE_RANDOM_SEED = HYBRID_RANDOM_SEED
CANDIDATE_RESCUE_F1_TOLERANCE = 0.005
RESCUE_FURNITURE_HEADS = {
	"bed", "bench", "cabinet", "chair", "couch", "counter", "crib", "desk",
	"dresser", "frame", "nightstand", "shelf", "sofa", "stool", "table",
}
RESCUE_GENERIC_LOCATIONS = PLACE_HEADS | {
	"building", "clinic", "facility", "field", "hallway", "mall", "office", "store", "yard",
}
RESCUE_BODY_PARTS = BODY_PARTS | {
	"fibula", "humerus", "nare", "radius", "tibia", "ulna",
}
RESCUE_NONOBJECT_HEADS = NOT_OBJECTS | {"attack", "pruritus"}
RESCUE_PART_HEADS = {
	"bar", "bumper", "clip", "door", "edge", "frame", "handle", "hook", "pane", "pedal",
}
RESCUE_MODIFIER_DEPS = {"acl", "amod", "compound", "nmod"}
RESCUE_GENERIC_MODIFIERS = {
	"big", "heavy", "large", "last", "lateral", "left", "little", "lower", "medial",
	"new", "old", "proximal", "right", "small", "upper",
}


def _modifier_reaches_root(token, root) -> bool:
	"""Check that a left modifier belongs to the same compact noun structure."""
	current = token
	for _ in range(5):
		if current.dep_ not in RESCUE_MODIFIER_DEPS:
			return False
		if current.head == root:
			return True
		if current.head.i <= current.i or current.head.i > root.i:
			return False
		current = current.head
	return False


def _candidate_modifier_allowed(token) -> bool:
	"""Keep identifying noun modifiers while excluding measurements and actors."""
	if not token.is_alpha or token.like_num or token.dep_ not in RESCUE_MODIFIER_DEPS:
		return False
	if token.lower_ in RESCUE_GENERIC_MODIFIERS:
		return False
	if token.lemma_ in PEOPLE | BODY_PARTS:
		return token.lemma_ in PRODUCT_MODIFIERS
	if token.lemma_ in NOT_OBJECTS:
		return token.lower_ in KEEP_MODIFIERS | IDENTIFYING_MODIFIERS
	return True


def expand_rescue_phrase(doc, root, lower_bound: int = 0):
	"""Return a conservative compound/modifier span ending at the noun head."""
	start = root.i
	for index in range(root.i - 1, lower_bound - 1, -1):
		token = doc[index]
		if token.is_punct and token.text in {"-", "/"}:
			start = index
			continue
		if not _candidate_modifier_allowed(token) or not _modifier_reaches_root(token, root):
			break
		start = index
	return doc[start:root.i + 1]


def _governing_verb(root):
	current = root
	for _ in range(6):
		if current.head == current:
			return None
		current = current.head
		if current.pos_ in {"VERB", "AUX"}:
			return current
	return None


def _nearest_action(root):
	actions = [
		token for token in root.doc
		if token.pos_ in {"VERB", "AUX"}
		and action_lemma(token) in INJURY_EVENT_ACTIONS | USE_ACTIONS
	]
	return min(actions, key=lambda token: abs(token.i - root.i)) if actions else None


def _rescue_phrase_text(span) -> str:
	phrase = " ".join(token.lower_ for token in span).strip()
	return re.sub(r"\s*([- /])\s*", r"\1", phrase)


def _make_rescue_candidate(doc, span, root, from_noun_chunk: bool, rule_keys: set) -> dict:
	phrase = _rescue_phrase_text(span)
	governing = _governing_verb(root)
	nearest = _nearest_action(root)
	preposition = root.head.lower_ if root.head.pos_ == "ADP" else ""
	head_lemma = root.lemma_.lower() if root.lemma_ else root.lower_
	anatomical_surface = head_lemma == "surface" and any(
		token.lower_ == "of"
		and token.i + 1 < len(doc)
		and doc[token.i + 1].lemma_ in RESCUE_BODY_PARTS
		for token in doc[root.i + 1:min(len(doc), root.i + 4)]
	)
	marked_text = (
		doc.text[:span.start_char]
		+ " [CANDIDATE] "
		+ doc.text[span.start_char:span.end_char]
		+ " [/CANDIDATE] "
		+ doc.text[span.end_char:]
	)
	return {
		"phrase": phrase,
		"key": _object_key(phrase),
		"start_char": span.start_char,
		"end_char": span.end_char,
		"model_text": " ".join(marked_text.lower().split()),
		"head": head_lemma,
		"dependency": root.dep_,
		"governing_verb": action_lemma(governing) if governing is not None else "",
		"nearest_action": action_lemma(nearest) if nearest is not None else "",
		"action_distance": abs(nearest.i - root.i) if nearest is not None else -1,
		"preposition": preposition,
		"from_noun_chunk": from_noun_chunk,
		"rule_rejected": _object_key(phrase) not in rule_keys,
		"token_count": len([token for token in span if not token.is_punct]),
		"compound_length": sum(token.dep_ == "compound" for token in span),
		"body_part": head_lemma in RESCUE_BODY_PARTS,
		"person": head_lemma in PEOPLE,
		"non_object": head_lemma in RESCUE_NONOBJECT_HEADS,
		"generic_location": head_lemma in RESCUE_GENERIC_LOCATIONS,
		"surface": head_lemma in SURFACE_STRUCTURE_HEADS,
		"furniture": head_lemma in RESCUE_FURNITURE_HEADS,
		"activity": head_lemma in AMBIGUOUS_ACTIVITIES,
		"medical_context": head_lemma in CLINICAL_GEAR | MEDICAL_TRANSPORT,
		"anatomical_surface": anatomical_surface,
		"absent": is_absent(root),
		"post_injury": is_post_injury_context(root, span),
	}


def generate_broad_candidates(doc) -> list:
	"""Generate noun candidates before the strict injury-relevance decision."""
	rule_keys = {_object_key(phrase) for phrase in extract_objects(doc)}
	chunks_by_root = {chunk.root.i: chunk for chunk in doc.noun_chunks}
	candidates = []
	seen = set()

	def add(span, root, from_noun_chunk):
		candidate = _make_rescue_candidate(doc, span, root, from_noun_chunk, rule_keys)
		identity = (candidate["start_char"], candidate["end_char"], candidate["key"])
		if candidate["key"] and identity not in seen:
			seen.add(identity)
			candidates.append(candidate)

	for root in doc:
		if root.pos_ not in {"NOUN", "PROPN"} or not root.is_alpha or len(root.text) < 2:
			continue
		chunk = chunks_by_root.get(root.i)
		lower_bound = chunk.start if chunk is not None else max(0, root.left_edge.i)
		expanded = expand_rescue_phrase(doc, root, lower_bound=lower_bound)
		add(expanded, root, chunk is not None)
		if expanded.start != root.i:
			add(doc[root.i:root.i + 1], root, chunk is not None)

		# The parser often tags the head of "soda can" as an auxiliary. Preserve
		# this existing project-specific parse repair in the broader reservoir.
		if root.i + 1 < len(doc):
			follower = doc[root.i + 1]
			if follower.lower_ in MISPARSED_NOUN_HEADS and follower.pos_ in {"AUX", "VERB"}:
				add(doc[root.i:root.i + 2], root, chunk is not None)
	return candidates


def build_candidate_rescue_examples(data: pd.DataFrame, nlp=None) -> list:
	"""Create grouped, conservatively labeled candidate examples per case."""
	if nlp is None:
		nlp = spacy.load("en_core_web_sm", disable=["ner"])
	texts = ["" if pd.isna(text) else str(text).lower() for text in data["incident_text"]]
	examples_by_case = []
	for row, doc in zip(data.itertuples(index=False), nlp.pipe(texts, batch_size=128)):
		gold_keys = {_object_key(phrase) for phrase in _gold_phrases(row.gold_objects)}
		candidates = generate_broad_candidates(doc)
		for candidate in candidates:
			# Deliberately avoid semantic aliases such as bike/bicycle during labeling.
			candidate["label"] = int(candidate["key"] in gold_keys)
			candidate["case_id"] = row.case_id
		examples_by_case.append(candidates)
	return examples_by_case


def _candidate_text_rows(records):
	return [record["model_text"] for record in records]


def _candidate_structured_rows(records):
	fields = (
		"head", "dependency", "governing_verb", "nearest_action", "preposition",
		"from_noun_chunk", "rule_rejected", "token_count", "compound_length",
		"body_part", "person", "non_object", "generic_location", "surface",
		"furniture", "activity", "medical_context", "absent", "post_injury",
		"anatomical_surface", "action_distance",
	)
	return [{field: record[field] for field in fields} for record in records]


def train_candidate_rescue_classifier(records, seed: int = CANDIDATE_RESCUE_RANDOM_SEED):
	"""Train the explicit TF-IDF plus logistic candidate relevance model."""
	from sklearn.feature_extraction import DictVectorizer
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.pipeline import FeatureUnion, Pipeline
	from sklearn.preprocessing import FunctionTransformer

	labels = [record["label"] for record in records]
	if set(labels) != {0, 1}:
		raise ValueError("Candidate training requires both positive and negative examples.")
	features = FeatureUnion(
		[
			(
				"context",
				Pipeline(
					[
						("select", FunctionTransformer(_candidate_text_rows, validate=False)),
						("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
					]
				),
			),
			(
				"structured",
				Pipeline(
					[
						("select", FunctionTransformer(_candidate_structured_rows, validate=False)),
						("encode", DictVectorizer()),
					]
				),
			),
		]
	)
	model = Pipeline(
		[
			("features", features),
			(
				"classifier",
				LogisticRegression(
					class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
				),
			),
		]
	)
	model.fit(records, labels)
	return model


def predict_candidate_rescue(model, records) -> list:
	"""Attach OBJECT_INVOLVED probabilities without changing candidate text."""
	if not records:
		return []
	probabilities = model.predict_proba(records)
	positive_index = list(model.named_steps["classifier"].classes_).index(1)
	return [
		{**record, "probability": float(probability[positive_index])}
		for record, probability in zip(records, probabilities)
	]


def _candidate_passes_safety(candidate) -> bool:
	"""Block obvious nonobjects without reusing the strict relevance filter."""
	prosthetic = "prosthetic" in candidate["phrase"].split()
	return not (
		candidate["person"]
		or candidate["non_object"]
		or candidate["generic_location"]
		or candidate["activity"]
		or candidate["anatomical_surface"]
		or candidate["absent"]
		or candidate["post_injury"]
		or (candidate["body_part"] and not prosthetic)
	)


def _contiguous_phrase(shorter: str, longer: str) -> bool:
	short_tokens = _object_key(shorter).split()
	long_tokens = _object_key(longer).split()
	return any(
		long_tokens[index:index + len(short_tokens)] == short_tokens
		for index in range(len(long_tokens) - len(short_tokens) + 1)
	)


def _same_object_variant(first: str, second: str) -> bool:
	"""Collapse only clear modifier or named-part variants of one object."""
	first_tokens = _object_key(first).split()
	second_tokens = _object_key(second).split()
	if not first_tokens or not second_tokens:
		return False
	shorter, longer = (first, second) if len(first_tokens) <= len(second_tokens) else (second, first)
	short_tokens = _object_key(shorter).split()
	long_tokens = _object_key(longer).split()
	if not _contiguous_phrase(shorter, longer):
		return False
	return short_tokens[-1] == long_tokens[-1] or long_tokens[-1] in RESCUE_PART_HEADS


def _deduplicate_object_phrases(objects) -> list:
	"""Prefer a longer identifying phrase only for clear lexical variants."""
	deduplicated = []
	for phrase in objects:
		variant_indices = [
			index for index, existing in enumerate(deduplicated)
			if _same_object_variant(existing, phrase)
		]
		if not variant_indices:
			deduplicated.append(phrase)
			continue
		longest_existing = max(
			(deduplicated[index] for index in variant_indices), key=lambda value: len(value.split())
		)
		if len(phrase.split()) <= len(longest_existing.split()):
			continue
		first_index = min(variant_indices)
		deduplicated = [
			existing for index, existing in enumerate(deduplicated) if index not in variant_indices
		]
		deduplicated.insert(first_index, phrase)
	return deduplicated


def combine_with_candidate_rescue(
	base_predictions,
	candidate_predictions,
	threshold: float = CANDIDATE_RESCUE_THRESHOLD,
) -> list:
	"""Add high-confidence candidates and deduplicate conservative phrase variants."""
	if len(base_predictions) != len(candidate_predictions):
		raise ValueError("Base and candidate prediction counts must match.")
	combined = []
	for base_cell, candidates in zip(base_predictions, candidate_predictions):
		objects = _deduplicate_object_phrases(
			[item.strip() for item in str(base_cell or "").split(";") if item.strip()]
		)
		accepted = sorted(
			(
				candidate for candidate in candidates
				if candidate["probability"] >= threshold and _candidate_passes_safety(candidate)
			),
			key=lambda candidate: (-candidate["token_count"], -candidate["probability"]),
		)
		accepted_spans = []
		for candidate in accepted:
			if any(
				candidate["start_char"] >= start and candidate["end_char"] <= end
				for start, end in accepted_spans
			):
				continue
			phrase = candidate["phrase"]
			key = _object_key(phrase)
			if not key or any(_object_key(existing) == key for existing in objects):
				continue
			variant_indices = [
				index for index, existing in enumerate(objects)
				if _same_object_variant(existing, phrase)
			]
			if variant_indices:
				longest_existing = max(
					(objects[index] for index in variant_indices), key=lambda value: len(value.split())
				)
				if len(phrase.split()) <= len(longest_existing.split()):
					continue
				objects = [
					existing for index, existing in enumerate(objects) if index not in variant_indices
				]
			objects.append(phrase)
			accepted_spans.append((candidate["start_char"], candidate["end_char"]))
		combined.append("; ".join(objects))
	return combined


def cross_validate_candidate_rescue(
	data: pd.DataFrame,
	folds=None,
	n_folds: int = HYBRID_CV_FOLDS,
	seed: int = CANDIDATE_RESCUE_RANDOM_SEED,
) -> dict:
	"""Produce candidate probabilities from models held out by case_id."""
	data = data.reset_index(drop=True).copy()
	if folds is None:
		folds = _case_folds(data, n_folds, seed)
	examples_by_case = build_candidate_rescue_examples(data)
	oof_predictions = [None] * len(data)
	all_indices = set(range(len(data)))
	for fold_number, held_out_indices in enumerate(folds):
		training_indices = sorted(all_indices - set(held_out_indices))
		training_records = [
			record for index in training_indices for record in examples_by_case[index]
		]
		model = train_candidate_rescue_classifier(training_records, seed=seed + fold_number)
		for index in held_out_indices:
			oof_predictions[index] = predict_candidate_rescue(model, examples_by_case[index])
	if any(predictions is None for predictions in oof_predictions):
		raise RuntimeError("Every case must receive candidate predictions from one held-out fold.")
	all_examples = [record for case in examples_by_case for record in case]
	return {
		"folds": folds,
		"predictions": oof_predictions,
		"positive_examples": sum(record["label"] for record in all_examples),
		"negative_examples": sum(not record["label"] for record in all_examples),
		"examples_by_case": examples_by_case,
	}


def _select_candidate_threshold(threshold_results: dict) -> float:
	best_f1 = max(item["metrics"]["micro_f1"] for item in threshold_results.values())
	competitive = [
		threshold for threshold, item in threshold_results.items()
		if best_f1 - item["metrics"]["micro_f1"] <= CANDIDATE_RESCUE_F1_TOLERANCE
	]
	return max(
		competitive,
		key=lambda threshold: (
			threshold_results[threshold]["metrics"]["precision"],
			threshold_results[threshold]["metrics"]["micro_f1"],
			threshold,
		),
	)


def _candidate_threshold_results(data, base_predictions, rescue_predictions) -> dict:
	results = {}
	for threshold in CANDIDATE_RESCUE_THRESHOLD_GRID:
		predictions = combine_with_candidate_rescue(
			base_predictions, rescue_predictions, threshold=threshold
		)
		results[threshold] = {
			"predictions": predictions,
			"metrics": _score_object_predictions(data["gold_objects"], predictions),
		}
	return results


def cross_validate_enhanced_hybrid(data: pd.DataFrame, span_result=None) -> dict:
	"""Compare rule, span, candidate, and combined out-of-fold recovery paths."""
	data = data.reset_index(drop=True).copy()
	if span_result is None:
		span_result = cross_validate_span_recovery(data)
	folds = _case_folds(data, HYBRID_CV_FOLDS, HYBRID_RANDOM_SEED)
	candidate_result = cross_validate_candidate_rescue(data, folds=folds)
	rule_predictions = span_result["rule_predictions"]
	span_predictions = span_result["hybrid_predictions"]

	candidate_only_results = _candidate_threshold_results(
		data, rule_predictions, candidate_result["predictions"]
	)
	candidate_only_threshold = _select_candidate_threshold(candidate_only_results)
	enhanced_results = _candidate_threshold_results(
		data, span_predictions, candidate_result["predictions"]
	)
	best_threshold = _select_candidate_threshold(enhanced_results)
	return {
		"rule_predictions": rule_predictions,
		"span_predictions": span_predictions,
		"candidate_predictions": candidate_result["predictions"],
		"candidate_only_threshold": candidate_only_threshold,
		"candidate_only_predictions": candidate_only_results[candidate_only_threshold]["predictions"],
		"candidate_only_metrics": candidate_only_results[candidate_only_threshold]["metrics"],
		"enhanced_threshold_results": enhanced_results,
		"best_threshold": best_threshold,
		"enhanced_predictions": enhanced_results[best_threshold]["predictions"],
		"enhanced_metrics": enhanced_results[best_threshold]["metrics"],
		"positive_examples": candidate_result["positive_examples"],
		"negative_examples": candidate_result["negative_examples"],
		"examples_by_case": candidate_result["examples_by_case"],
	}


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
