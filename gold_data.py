"""Small hand-labelled test set (30 sentences, 10 ambiguous nouns) plus demo text.

Gold labels are WordNet 3.0 synset names chosen by the author of this project.
It is intentionally small: use it for a quick sanity check, not as a benchmark.
"""

N = "n"
GOLD = [
    # bank
    ("She deposited her paycheck at the bank on Friday.", "bank", N, "depository_financial_institution.n.01"),
    ("The bank approved our mortgage loan after reviewing our savings.", "bank", N, "depository_financial_institution.n.01"),
    ("We had a picnic on the grassy bank of the river and watched the water.", "bank", N, "bank.n.01"),
    # bass
    ("He turned up the bass so the music had a deeper sound.", "bass", N, "bass.n.01"),
    ("We caught a large bass in the lake and grilled the fish for dinner.", "bass", N, "bass.n.08"),
    ("The choir needed a strong bass, a male singer with a very low voice.", "bass", N, "bass.n.03"),
    # bat
    ("A bat flew out of the cave at night, hunting insects with its wings.", "bat", N, "bat.n.01"),
    ("The batter swung the bat and hit the baseball over the fence.", "bat", N, "bat.n.05"),
    ("The mammal hung upside down like a bat in the dark attic.", "bat", N, "bat.n.01"),
    # plant
    ("The car company opened a new plant where workers assemble engines in a factory.", "plant", N, "plant.n.01"),
    ("Water the plant daily so its leaves and roots get enough sunlight and moisture.", "plant", N, "plant.n.02"),
    ("The power plant produces electricity for the industrial city.", "plant", N, "plant.n.01"),
    # spring
    ("The garden blooms in spring after the long, cold winter.", "spring", N, "spring.n.01"),
    ("The mattress has a metal spring that bounces back when you press it.", "spring", N, "spring.n.02"),
    ("Fresh water flows from the spring in the mountain valley.", "spring", N, "spring.n.03"),
    # mouse
    ("The cat chased a small mouse, a rodent with a long tail, across the kitchen.", "mouse", N, "mouse.n.01"),
    ("Click the left button of the mouse to select the file on the computer screen.", "mouse", N, "mouse.n.04"),
    ("The mouse nibbled the cheese and hid in a hole in the wall.", "mouse", N, "mouse.n.01"),
    # crane
    ("The construction crane lifted a heavy steel beam to the top of the building.", "crane", N, "crane.n.04"),
    ("A tall crane waded in the marsh, a long-necked bird looking for fish.", "crane", N, "crane.n.05"),
    ("The crane stood on one leg near the pond and spread its wings.", "crane", N, "crane.n.05"),
    # match
    ("The two teams played a tense football match, and the contest ended in a draw.", "match", N, "match.n.02"),
    ("He struck a match to light the candle and the flame flickered.", "match", N, "match.n.01"),
    ("The tennis player won the match in straight sets against her rival.", "match", N, "match.n.02"),
    # interest
    ("The bank charges five percent interest on the loan, calculated on the money borrowed.", "interest", N, "interest.n.04"),
    ("She has a deep interest in astronomy and reads about the stars with curiosity.", "interest", N, "interest.n.01"),
    ("Savings accounts pay interest on the deposit every year.", "interest", N, "interest.n.04"),
    # light
    ("Turn on the light, the room is too dark to read.", "light", N, "light.n.02"),
    ("Light travels as electromagnetic radiation, and the speed of light is constant.", "light", N, "light.n.01"),
    ("The lamp gave a warm light and the bulb illuminated the table.", "light", N, "light.n.02"),
]

GOLD_ITEMS = [
    {"sentence": s, "target": t, "pos": p, "gold_synset": g} for (s, t, p, g) in GOLD
]

DEMO_TEXT = (
    "He sat on the bank of the river and watched the water flow. "
    "She went to the bank to deposit money into her savings account. "
    "The bass guitar produced a deep sound during the concert. "
    "The fisherman caught a huge bass in the lake. "
    "The factory plant employs hundreds of workers. "
    "Water the plant every morning so the leaves stay green. "
    "Move the mouse and click the button on the screen. "
    "A mouse ran across the floor and the cat chased it. "
    "The crane lifted the steel beams at the construction site. "
    "A white crane stood quietly in the marsh near the pond."
)
