import secrets
mysticism = [
    'Your real strength comes from being the best "You" you can be. [Po]',
    'The hardcore do understand. [Tigress]',
    "I'm not A big fat panda, I'm The big fat panda. [Po]",
    'If you only do what you can do, you will never be more than what you are now. [Master Shifu]',
    'There are no coincidences in this world. [Master Oogway]',
    "Yesterday is history, tomorrow is a mystery, but today is a gift. That's why it's called the present. [Master Oogway]",
    "There is no secret ingredient. It's just you. [Po]",
    'Your mind is like this water, my friend. When it is agitated, it becomes difficult to see. But if you allow it to settle, the answer becomes clear. [Master Oogway]',
    'You must let go of the illusion of control. [Master Oogway]',
    'We do not wash our pits in the pool of sacred tears. [Master Shifu]',
    'Legend tells of a legendary warrior whose kung fu skills were the stuff of legend. [Po]',
    'Skadoosh! [Po]',
    "The strongest of us sometimes have the hardest time fighting what's on the inside. [Po]",
    'One often meets his destiny on the road he takes to avoid it. [Master Oogway]',
    "Let's not forget what happened to the man who suddenly got everything he wanted. [Po]",
    'There is no charge for awesome. [Po]',
    'The journey of a thousand miles begins with one step. ',
    'A wise man can learn more from a foolish question than a fool can learn from a wise answer. [Master Shifu]'
]

def oracle_word():
    return secrets.choice(mysticism)

def ending_sentence(verbose: bool):
    # Define a function to print if verbose
    verboseprint = print if verbose else lambda *a, **k: None
    verboseprint(oracle_word())