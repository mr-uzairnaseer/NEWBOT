import re
import random
import logging

logger = logging.getLogger(__name__)

# Phonetic corrections for Kaldi/Vosk STT output
_PHONETIC_CORRECTIONS = [
    (r"\bsick tea\b", "sixty"),
    (r"\bsick steep\b", "sixty"),
    (r"\bsick t\b", "sixty"),
    (r"\bfor tea\b", "forty"),
    (r"\bfor t\b", "forty"),
    (r"\bfif tea\b", "fifty"),
    (r"\bfif t\b", "fifty"),
    (r"\bthir tea\b", "thirty"),
    (r"\bthir t\b", "thirty"),
    (r"\bsev entire\b", "seventy"),
    (r"\bsev in tea\b", "seventy"),
    (r"\bsev inti\b", "seventy"),
    (r"\baid tea\b", "eighty"),
    (r"\bate tea\b", "eighty"),
    (r"\bnine tea\b", "ninety"),
    (r"\btwenty (one|two|three|four|five|six|seven|eight|nine)\b", r"twenty \1"),
    (r"\bya\b", "yeah"),
    (r"\byah\b", "yeah"),
    (r"\bnah\b", "no"),
    (r"\byeah sure\b", "yes"),
    (r"\bhow low\b", "hello"),
    (r"\bhallow\b", "hello"),
    (r"\bgood buy\b", "goodbye"),
    (r"\bthanks you\b", "thank you"),
]
_COMPILED_CORRECTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in _PHONETIC_CORRECTIONS]

def local_correct_stt(raw_text: str) -> str:
    """Instant phonetic correction — no API call."""
    corrected = raw_text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    if corrected != raw_text:
        logger.info(f"STT local fix: '{raw_text}' → '{corrected}'")
    return corrected

def check_keyword(text: str, keywords: list[str]) -> bool:
    """Helper to check if any of the keywords are in the text as whole words."""
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

# Reassurance / Empathy phrasing options for highly organic human-like rotation
_REASSURANCE_PHRASES = [
    "Got it.", "No worries at all.", "Totally fine.", "No problem.", 
    "Understood.", "Perfect.", "No worries.", "No problem at all."
]
_EMPATHY_PHRASES = [
    "I completely understand.", "I hear you loud and clear.", 
    "I completely get where you're coming from.", "That is totally understandable.", 
    "I understand completely.", "I completely respect that.", "I understand.", "Makes perfect sense."
]
_PRIVACY_PHRASES = [
    "I completely respect your privacy.", "Security and privacy are absolutely top priority.", 
    "I completely understand your caution.", "We definitely respect your space and privacy.", 
    "Safety is number one.", "I respect your privacy completely.", "I completely understand your safety concerns."
]
_RESPECT_PHRASES = [
    "I completely respect that.", "I absolutely respect your decision.", "Fair enough.", 
    "I completely respect where you're coming from.", "I hear you and absolutely respect that."
]
_PRESSURE_FREE_PHRASES = [
    "No pressure at all.", "Absolutely no pressure.", "Totally up to you.", 
    "No worries whatsoever.", "There is absolutely no rush or pressure."
]

def get_comfort_phrase(category: str, reduce_freq: bool = True) -> str:
    """
    Returns a comfort phrase from the specified category.
    Includes a random empty-return chance to prevent conversational clutter.
    """
    if reduce_freq and random.random() < 0.35:
        return ""
    
    if category == "reassurance":
        phrase = random.choice(_REASSURANCE_PHRASES)
    elif category == "empathy":
        phrase = random.choice(_EMPATHY_PHRASES)
    elif category == "privacy":
        phrase = random.choice(_PRIVACY_PHRASES)
    elif category == "respect":
        phrase = random.choice(_RESPECT_PHRASES)
    elif category == "pressure":
        phrase = random.choice(_PRESSURE_FREE_PHRASES)
    else:
        phrase = ""
        
    return phrase + " " if phrase else ""

def determine_active_step(assistant_msgs: list[str]) -> int:
    """
    Scans assistant messages history from most recent to oldest
    to find the active script step based on the question content.
    Returns 1-based step index. Defaults to 1 (Greeting & Q1) if none matched.
    """
    for msg in reversed(assistant_msgs):
        msg_lower = msg.lower()
        if any(phrase in msg_lower for phrase in ["how old", "age", "old are you", "age group", "over sixty", "over the age"]):
            return 2
        elif any(phrase in msg_lower for phrase in ["part a", "part a and b", "hello", "active right now", "red, white, and blue"]):
            return 1
    return 1

def text_to_number(text: str) -> int | None:
    """Converts a phrase containing English number words into an integer if possible."""
    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    
    # Clean up input text
    tokens = re.findall(r"\w+", text.lower())
    
    # Check if there is an explicit digit first
    digits = re.findall(r"\d+", text)
    if digits:
        return int(digits[0])
        
    # Otherwise, try to find word-based numbers
    total = 0
    found = False
    
    for token in tokens:
        if token in words:
            found = True
            val = words[token]
            # Handle tens followed by ones (e.g. fifty + eight = 58)
            if val >= 20 and total > 0 and total < 20:
                total += val
            elif val < 10 and total >= 20 and total % 10 == 0:
                # E.g. sixty (60) + five (5) = 65
                total += val
            else:
                total += val
                
    return total if found else None

def get_local_router_response(text: str, conversation_history: list[dict]) -> str | None:
    """
    Evaluates incoming user text against known intents, objections,
    and conversational contexts, then returns the correct, dynamic response phrase.
    Returns None if no high-confidence local intent matches (enabling LLM fallback).
    """
    assistant_msgs = [m["content"] for m in conversation_history if m["role"] == "assistant"]
    active_step = determine_active_step(assistant_msgs)
    last_user_msg = text.lower().strip()

    yes_keywords = ["yes", "yeah", "yep", "sure", "correct", "right", "i do", "ok", "okay", "think so", "believe so", "i think so", "i believe so"]
    no_keywords = ["no", "dont", "don't", "nope", "nah", "nevermind", "not really", "not active"]
    unsure_keywords = ["don't know", "dont know", "not sure", "no idea", "maybe", "not certain", "uncertain"]

    is_unsure = any(w in last_user_msg for w in unsure_keywords)
    is_yes = (check_keyword(last_user_msg, yes_keywords) or any(w in last_user_msg for w in ["yes", "yeah", "yep", "sure", "correct", "ok", "okay", "think so", "believe so"])) and not is_unsure
    is_no = (check_keyword(last_user_msg, no_keywords) or "don't" in last_user_msg or "dont" in last_user_msg or "no" in last_user_msg.split()) and not is_unsure

    age_keywords = [
        "twenty", "thirty", "forty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "old", "years"
    ]
    has_age = any(char.isdigit() for char in last_user_msg) or any(w in last_user_msg.lower() for w in age_keywords)

    is_dnc = any(w in last_user_msg for w in ["stop calling", "dont call", "don't call", "do not call", "remove me", "please stop", "bothering", "take me off", "wrong number", "not interested", "no thank you", "no thanks", "hang up", "bye"])
    is_identity = any(w in last_user_msg for w in ["who are you", "who is this", "what is your name", "your name", "what company", "who is calling", "who is emily", "i don't know you", "dont know you", "what is this company", "who do you work for"])
    is_why = any(w in last_user_msg for w in [
        "why are you calling", "why calling", "what is this about", "what is the reason", 
        "why do you want", "reason for this", "why are you asking", "why ask", "why do you ask", 
        "why do you need", "why need", "why are you checking", "why check", "why do you check", "why you asking",
        "why should i", "why do i have to", "why do you want to know", "why do you need to know", "why should i tell you"
    ])
    is_scam = (
        check_keyword(last_user_msg, ["scam", "fake", "legit", "robot", "artificial", "machine", "computer", "ai"])
        or any(w in last_user_msg for w in ["real person", "are you a robot"])
    )
    is_benefits = any(w in last_user_msg for w in ["benefit", "benefits", "what do you offer", "what do i get", "what are they", "what benefit", "what benefits", "qualify for", "what do i qualify", "food card", "flex card", "cash back"])
    is_ssn = check_keyword(last_user_msg, ["ssn"]) or "social security" in last_user_msg
    is_privacy_refusal = any(w in last_user_msg for w in [
        "not willing to", "not telling", "won't tell", "wont tell", "won't say", "wont say",
        "prefer not to", "private", "personal", "none of your business", "not sharing", 
        "confidential", "keep that to myself", "under lock and key", "won't share", "wont share", "willing to tell",
        "not going to tell", "not going to say", "wont give", "won't give"
    ])
    is_spouse = any(w in last_user_msg for w in ["spouse", "wife", "husband", "partner", "marry", "married", "significant other", "deals with", "handles"])
    is_location = any(w in last_user_msg for w in [
        "where are you calling from", "where is your office", "where are you located", "where are you based", 
        "where are you residing", "where do you live", "where are you", "location", "address"
    ])
    is_part_ab_expl = (
        any(w in last_user_msg for w in ["part a", "part b", "both parts"])
        and any(w in last_user_msg for w in ["explain", "tell me", "what is", "what are", "what do", "what covered", "mean", "meaning", "difference", "why"])
    )
    is_greeting = any(w == last_user_msg.strip() for w in ["hello", "hi", "hey", "hello emily", "hi emily", "hey emily", "good morning", "good afternoon"])
    is_bot_age = (
        any(w in last_user_msg for w in ["young", "pretty young", "so young", "sound young", "sounds young", "minor"])
        or (("how old" in last_user_msg or "what is your age" in last_user_msg or "what's your age" in last_user_msg) and ("you" in last_user_msg or "your" in last_user_msg or "emily" in last_user_msg))
    )
    is_clarification = any(w in last_user_msg for w in ["understand", "repeat", "hear", "what did you say", "say again", "what was that", "pardon", "what do you mean", "slow down"]) or last_user_msg in ["what", "who", "huh", "pardon"]

    # Calculate dynamic attempts from assistant messages
    step_1_attempts = sum(1 for m in assistant_msgs if any(w in m.lower() for w in ["part a", "part a and b", "hello", "active right now"]))
    step_2_attempts = sum(1 for m in assistant_msgs if any(w in m.lower() for w in ["old are you", "age group", "over sixty", "over the age"]))

    def get_step_1_question() -> str:
        # Check what was asked in the last assistant message
        last_assistant_msg = ""
        for m in reversed(conversation_history):
            if m["role"] == "assistant":
                last_assistant_msg = m["content"].lower()
                break
                
        # List of question options
        q1 = "Do you have Medicare Part A & B?"
        q2 = "Are both your Part A and Part B active right now?"
        q3 = "Do you have the red, white, and blue Medicare card handy?"
        
        # If we haven't asked anything yet
        if not last_assistant_msg:
            return q1
            
        # Avoid repeating what we just asked
        if "red, white, and blue" in last_assistant_msg or "blue medicare card" in last_assistant_msg:
            return q2
        elif "active right now" in last_assistant_msg:
            return q3
        elif "part a & b" in last_assistant_msg or "part a and b" in last_assistant_msg:
            return q2
            
        # Fallback to attempt count
        if step_1_attempts == 0:
            return q1
        elif step_1_attempts == 1:
            return q2
        else:
            return q3

    def get_step_2_question() -> str:
        # Check what was asked in the last assistant message
        last_assistant_msg = ""
        for m in reversed(conversation_history):
            if m["role"] == "assistant":
                last_assistant_msg = m["content"].lower()
                break
                
        # List of question options
        q1 = "How old are you right now?"
        q2 = "What is your age group right now?"
        q3 = "Are you generally over the age of sixty?"
        
        # If we haven't asked anything yet
        if not last_assistant_msg:
            return q1
            
        # Avoid repeating what we just asked
        if "over sixty" in last_assistant_msg or "over the age" in last_assistant_msg:
            return q2
        elif "age group" in last_assistant_msg:
            return q3
        elif "how old" in last_assistant_msg or "old are you" in last_assistant_msg:
            return q2
            
        # Fallback to attempt count
        if step_2_attempts == 0:
            return q1
        elif step_2_attempts == 1:
            return q2
        else:
            return q3

    # Intent-based dialogue trees
    if is_dnc:
        return "I am so sorry about that! I will definitely note down to remove your number. Have a wonderful day. [DROP]"
    
    elif is_greeting:
        if len(assistant_msgs) <= 1:
            return "Hi there! Yes, this is Emily calling from low insurance cost Medicare. Do you have Medicare Part A and B active?"
        else:
            return "Yes, I am still here! " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_bot_age:
        reassurance = "Haha, I get that a lot! I'm twenty-four, but I promise I'm fully qualified. "
        return reassurance + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_spouse:
        reassurance = "Ah, got it! That makes complete sense. If they are around, they can verify it, but just to check for your own eligibility, "
        return reassurance + (get_step_1_question() if active_step == 1 else get_step_2_question())

    elif is_identity:
        already_said = sum(1 for m in assistant_msgs if any(w in m.lower() for w in ["my name is emily", "calling from low insurance"]))
        if already_said == 0:
            reassurance = get_comfort_phrase("privacy") + "My name is Emily calling from low insurance cost Medicare. We are simply helping seniors review their Medicare options. "
        elif already_said == 1:
            reassurance = get_comfort_phrase("reassurance") + "Like I mentioned, my name is Emily and I am calling from low insurance cost Medicare. We are just helping seniors check their eligibility. "
        else:
            reassurance = get_comfort_phrase("respect") + "I am just a representative from low insurance cost Medicare checking basic eligibility so we don't waste your time. "
        return reassurance + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_location:
        return get_comfort_phrase("respect") + "We are based in houston, Texas, calling from low insurance cost Medicare. We are simply helping seniors review their Medicare options. " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_part_ab_expl:
        return get_comfort_phrase("reassurance") + "Part A covers hospital stays, and Part B covers doctor visits. They make up standard Medicare. " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_why:
        if active_step == 2:
            return get_comfort_phrase("reassurance") + "We ask because these additional premium benefits are age-dependent. " + get_step_2_question()
        
        already_said = sum(1 for m in assistant_msgs if any(w in m.lower() for w in ["reaching out to local seniors", "reason we are calling"]))
        if already_said == 0:
            reassurance = get_comfort_phrase("reassurance") + "We are reaching out to local seniors to help them review if they are eligible for additional benefits like dental, vision, hearing, and food card allowances. "
        else:
            reassurance = get_comfort_phrase("respect") + "We are just checking local seniors' eligibility so they don't miss out on premium benefits like flex cards or cash back. "
        return reassurance + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_scam:
        return get_comfort_phrase("privacy") + "No, I am a live representative calling from low insurance cost Medicare. We never ask for any private SSN numbers on this call. We just want to check. " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_benefits:
        already_said = sum(1 for m in assistant_msgs if any(w in m.lower() for w in ["food card", "flex card", "cash back"]))
        if already_said == 0:
            reassurance = get_comfort_phrase("reassurance") + "We check for allowances like the food card, three hundred dollars cash back, flex cards, and very low premiums. "
        else:
            reassurance = get_comfort_phrase("respect") + "Those are premium benefits that add on top of standard Medicare to cover dental, vision, and groceries. "
        return reassurance + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_ssn:
        return get_comfort_phrase("privacy") + "You don't need to give it to me! You can keep your card handy and verify it securely in a moment. " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    elif is_privacy_refusal:
        if active_step == 1:
            return get_comfort_phrase("privacy") + "We simply help seniors check their basic eligibility first. " + get_step_1_question()
        else:
            return get_comfort_phrase("privacy") + "We ask because these additional premium benefits are age-dependent. " + get_step_2_question()
        
    elif is_unsure:
        if active_step == 1:
            if step_1_attempts <= 1:
                return get_comfort_phrase("empathy") + "If you are sixty-five or older, you usually have Part A and B active. Do you receive those benefits, or have a red, white, and blue card?"
            elif step_1_attempts == 2:
                return get_comfort_phrase("reassurance") + "If you visit a doctor, is that covered by standard Medicare? That usually means both parts are active."
            else:
                return get_comfort_phrase("respect") + "Let's do this—I can get a specialist on the line who can quickly verify that. How old are you right now?"
        else:
            if step_2_attempts <= 1:
                return get_comfort_phrase("empathy") + "We ask because eligibility is based on age. Are you sixty or older right now?"
            else:
                return get_comfort_phrase("respect") + "Let's get that specialist on the line right away to verify. [TRANSFER]"
                
    elif is_clarification:
        return get_comfort_phrase("reassurance") + "Let me repeat. " + (get_step_1_question() if active_step == 1 else get_step_2_question())
        
    else:
        if active_step == 1:
            if is_yes:
                return get_comfort_phrase("reassurance") + get_step_2_question()
            elif is_no:
                return get_comfort_phrase("respect") + "You need Medicare Part A and B to qualify. Have a wonderful day! [DROP]"
            else:
                # Evasive/out-of-scope response: return None to allow LLM fallback
                return None
        else:
            age_val = text_to_number(last_user_msg)
            
            # Ultra-robust age validation to avoid false positive matches on money, time, or negation
            is_valid_age = False
            if age_val is not None:
                exclusions = ["dollar", "dollars", "buck", "bucks", "cent", "cents", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "percent", "mile", "miles", "o'clock", "degree", "degrees"]
                has_exclusion = any(w in last_user_msg for w in exclusions)
                has_negation = any(w in last_user_msg for w in ["not", "dont", "don't", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt"])
                
                is_plausible_age = True
                if age_val < 45:
                    is_plausible_age = any(w in last_user_msg for w in ["years", "old", "age", "birthday", "yo"]) or any(char.isdigit() for char in last_user_msg)
                
                if is_plausible_age and not has_exclusion and not has_negation:
                    is_valid_age = True

            if is_valid_age:
                if age_val >= 60:
                    return get_comfort_phrase("reassurance") + "Let me get that specialist on the line for you right away. [TRANSFER]"
                else:
                    return get_comfort_phrase("respect") + "Unfortunately, you must be sixty or older to qualify. Have a wonderful day! [DROP]"
            elif is_no:
                return get_comfort_phrase("respect") + "Unfortunately, you must be sixty or older to qualify. Have a wonderful day! [DROP]"
            elif (is_yes and step_2_attempts >= 2) or (has_age and any(w in last_user_msg for w in ["over sixty", "over 60", "older than sixty", "older than 60"]) and not any(w in last_user_msg for w in ["not", "dont", "don't"])):
                return get_comfort_phrase("reassurance") + "Let me get that specialist on the line for you right away. [TRANSFER]"
            else:
                # Evasive/out-of-scope age response: return None to allow LLM fallback
                return None
