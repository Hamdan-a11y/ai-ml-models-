"""
Spam Email Classifier
=====================
Uses a Naive Bayes algorithm (trained on built-in sample data) to classify
emails as SPAM or HAM (not spam).

Usage:
    python spam_classifier.py                  # run built-in demo
    python spam_classifier.py --interactive    # classify your own emails
"""

import re
import math
import argparse
from collections import defaultdict


# ─────────────────────────────────────────────
#  Training Data
# ─────────────────────────────────────────────

TRAINING_DATA = [
    # (label, email_text)
    # --- SPAM ---
    ("spam", "Congratulations! You have won a $1,000,000 lottery prize. Click here to claim now!"),
    ("spam", "FREE offer! Buy now and get 50% discount. Limited time only. Act fast!"),
    ("spam", "You are selected for a special prize. Send your bank details to claim your reward."),
    ("spam", "Earn money fast working from home. No experience needed. $500 daily guaranteed!"),
    ("spam", "URGENT: Your account will be suspended. Verify your details immediately by clicking this link."),
    ("spam", "Hot singles in your area want to meet you tonight. Click here for free access."),
    ("spam", "Buy cheap Viagra online. No prescription needed. Lowest price guaranteed!"),
    ("spam", "You have been pre-approved for a $50,000 loan. Apply now, no credit check required."),
    ("spam", "Win a free iPhone 15! Just fill in your details and claim your prize today!"),
    ("spam", "MAKE MONEY ONLINE! Thousands of people are earning $3000 a week from home!"),
    ("spam", "Dear lucky winner, you have been selected from millions to receive our grand prize."),
    ("spam", "Special promotion: enlarge your business profits with our secret investment formula."),
    ("spam", "Your PayPal account has been compromised. Click here to verify your identity now."),
    ("spam", "Lose 30 pounds in 30 days with this miracle pill doctors don't want you to know about."),
    ("spam", "Exclusive deal for you only! Investment opportunity with 200% guaranteed returns."),
    # --- HAM ---
    ("ham", "Hi John, just wanted to confirm our meeting tomorrow at 10am. Let me know if that works."),
    ("ham", "Please find attached the project report for Q3. Let me know if you have any feedback."),
    ("ham", "Hey, are you coming to the team lunch on Friday? We are going to the new Italian place."),
    ("ham", "Reminder: your dentist appointment is scheduled for Monday at 3pm."),
    ("ham", "Thanks for sending over the documents. I will review them and get back to you by end of day."),
    ("ham", "The meeting notes from yesterday have been uploaded to the shared drive."),
    ("ham", "Can you please review the pull request I opened this morning? It fixes the login bug."),
    ("ham", "Happy birthday! Hope you have a wonderful day surrounded by family and friends."),
    ("ham", "Your Amazon order #302-1234567 has been shipped and will arrive by Thursday."),
    ("ham", "Just checking in — how is the new project coming along? Need any help from my end?"),
    ("ham", "The quarterly budget report is due by next Friday. Please submit your department figures."),
    ("ham", "Hi, I wanted to follow up on our conversation from last week regarding the proposal."),
    ("ham", "Class is cancelled tomorrow due to the university holiday. See you all next Monday."),
    ("ham", "Dinner is at 7pm tonight. Mom is making your favourite pasta, don't be late!"),
    ("ham", "Your subscription to Netflix has been renewed for the next month. Thank you."),
]


# ─────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


# ─────────────────────────────────────────────
#  Naive Bayes Classifier
# ─────────────────────────────────────────────

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes with Laplace (add-1) smoothing.

    P(spam | email) ∝ P(spam) * ∏ P(word | spam)
    P(ham  | email) ∝ P(ham)  * ∏ P(word | ham)
    """

    def __init__(self):
        self.class_word_counts: dict[str, defaultdict] = {}
        self.class_doc_counts: dict[str, int] = {}
        self.class_total_words: dict[str, int] = {}
        self.vocabulary: set[str] = set()
        self.total_docs: int = 0

    def train(self, data: list[tuple[str, str]]):
        for label, text in data:
            words = tokenize(text)
            if label not in self.class_word_counts:
                self.class_word_counts[label] = defaultdict(int)
                self.class_doc_counts[label] = 0
                self.class_total_words[label] = 0

            self.class_doc_counts[label] += 1
            self.class_total_words[label] += len(words)
            for word in words:
                self.class_word_counts[label][word] += 1
                self.vocabulary.add(word)

        self.total_docs = len(data)

    def _log_probability(self, words: list[str], label: str) -> float:
        """Log P(label) + sum of log P(word | label) with Laplace smoothing."""
        vocab_size = len(self.vocabulary)

        # Prior: log P(label)
        log_prob = math.log(self.class_doc_counts[label] / self.total_docs)

        # Likelihood: log P(word | label)
        total_words = self.class_total_words[label]
        for word in words:
            word_count = self.class_word_counts[label].get(word, 0)
            # Laplace smoothing: add 1 to numerator, vocab_size to denominator
            log_prob += math.log((word_count + 1) / (total_words + vocab_size))

        return log_prob

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Returns (predicted_label, confidence_scores).
        Confidence is converted from log-probs to percentages.
        """
        words = tokenize(text)
        scores = {
            label: self._log_probability(words, label)
            for label in self.class_word_counts
        }

        # Convert log-probs to normalised probabilities
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {k: round(v / total * 100, 2) for k, v in exp_scores.items()}

        predicted = max(scores, key=scores.get)
        return predicted, probabilities


# ─────────────────────────────────────────────
#  Pretty Printing
# ─────────────────────────────────────────────

def print_result(email: str, predicted: str, probs: dict, label: str = None):
    bar_len = 40
    spam_pct = probs.get("spam", 0)
    ham_pct  = probs.get("ham", 0)

    spam_bar = "█" * int(spam_len := spam_pct / 100 * bar_len)
    ham_bar  = "█" * int(ham_pct  / 100 * bar_len)

    verdict = "🚨 SPAM" if predicted == "spam" else "✅ HAM (Not Spam)"
    if label:
        correct = "✔ Correct" if predicted == label else "✘ Wrong"
        actual = f"  Actual: {label.upper()}  {correct}"
    else:
        actual = ""

    preview = email if len(email) <= 80 else email[:77] + "..."

    print("─" * 60)
    print(f"  Email : {preview}")
    print(f"  Result: {verdict}{actual}")
    print(f"  SPAM  [{spam_bar:<{bar_len}}] {spam_pct:5.1f}%")
    print(f"  HAM   [{ham_bar:<{bar_len}}] {ham_pct:5.1f}%")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run_demo(clf: NaiveBayesClassifier):
    test_emails = [
        ("spam", "You have won a free vacation! Claim your prize by sending your bank information now."),
        ("ham",  "Hi Sarah, please review the attached slides before our meeting tomorrow morning."),
        ("spam", "Make $5000 a week from home! No experience needed. Click here to start earning today."),
        ("ham",  "Your package has been delivered to the front door. Thanks for shopping with us."),
        ("spam", "URGENT: Click this link immediately to avoid your account being permanently deleted."),
        ("ham",  "Hey, just wanted to check if you got my last email about the project deadline."),
    ]

    print("\n" + "=" * 60)
    print("   SPAM CLASSIFIER — DEMO")
    print("=" * 60)
    print(f"   Training samples : {len(TRAINING_DATA)}")
    print(f"   Vocabulary size  : {len(clf.vocabulary)} words")
    print(f"   Testing on       : {len(test_emails)} emails")
    print("=" * 60)

    correct = 0
    for label, email in test_emails:
        predicted, probs = clf.classify(email)
        print_result(email, predicted, probs, label)
        if predicted == label:
            correct += 1

    accuracy = correct / len(test_emails) * 100
    print("─" * 60)
    print(f"\n  Accuracy on test set: {correct}/{len(test_emails)}  ({accuracy:.0f}%)\n")


def run_interactive(clf: NaiveBayesClassifier):
    print("\n" + "=" * 60)
    print("   SPAM CLASSIFIER — INTERACTIVE MODE")
    print("   Type an email and press Enter to classify.")
    print("   Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            email = input("\nEmail text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if email.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not email:
            print("  (empty input, try again)")
            continue

        predicted, probs = clf.classify(email)
        print_result(email, predicted, probs)


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes Spam Classifier")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive mode to classify your own emails"
    )
    args = parser.parse_args()

    # Train
    clf = NaiveBayesClassifier()
    clf.train(TRAINING_DATA)

    if args.interactive:
        run_interactive(clf)
    else:
        run_demo(clf)


if __name__ == "__main__":
    main()
