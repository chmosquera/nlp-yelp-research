
##
# First, you're going to need to import wordnet:
from nltk.corpus import wordnet
def synonymExercise(word):
    # Then, we're going to use the term "program" to find synsets like so:
    syns = wordnet.synsets(word)

    # An example of a synset:
    print("name: ", syns[0].name())

    # Just the word:
    print("lemma: ", syns[0].lemmas()[0].name())

    # Definition of that first synset:
    print("definition: ", syns[0].definition())

    # Examples of the word in use in sentences:
    print("examples: ", syns[0].examples())

# synonymExercise("food")

##
# First, you're going to need to import wordnet:
from nltk.corpus import wordnet
def synonymExercise2(words):
    synonyms = {}
    antonyms = {}
    definitions = {}

    for word in words:
        synonyms[word] = []
        antonyms[word] = []
        definitions[word] = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms[word].append(l.name())
                if l.antonyms():
                    antonyms[word].append(l.antonyms()[0].name())
        print(set(synonyms))
        print(set(antonyms))

    return synonyms, antonyms

##
from nltk.corpus import wordnet
def foodDictionary():
    path = "logs/foods.txt"
    f = open(path, "a+")
    root = "food"

    queue = [root]

    # Keep track of the words I've already included and don't include again
    history = []
    while len(queue) > 0:
        word = queue.pop()

        # Sometimes duplicates are in the queue, skip them if we already did it
        if word in history:
            continue

        # Otherwise this is a new word, save to history
        history.append(word)
        syns = wordnet.synsets(word)

        # Save the current word
        f.write(syns[0].lemmas()[0].name() + ",")

        for syn in syns:
            # Extract all other lemmas
            lems = [lem.name() for lem in syn.lemmas()[1:]]

            # Decide whether you want to include these lemmas
            if len(lems) > 0:
                print(lems, ": ", syn.definition())
                print("Do you want to include these into the food log? (y if yes)")
                log_this = input()

                # If 'y' then write into file
                if log_this == 'y':
                    for lem in lems:
                        if lem not in history:
                            f.write(lem + ",")
                            queue.append(lem)

                # If exit, close the file and stop this func
                if log_this == "exit!":
                    f.close()
                    return



foodDictionary()