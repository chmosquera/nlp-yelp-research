
from nltk.corpus import wordnet
def createDictionary(words, path="logs/extractingTopics_temp.txt"):
    f = open(path, "a+")

    queue = []
    # Keep track of the words I've already included and don't include again
    history = []
    for word in words:
        syns = wordnet.synsets(word)
        f.write(word + ",")
        for syn in syns:
            queue.append(syn)

    while len(queue) > 0:
        syn = queue.pop(0)

        # Sometimes duplicates are in the queue, skip them if we already did it
        if syn in history:
            continue
        # Otherwise this is a new word, save to history
        history.append(syn)

        # Lemmas can help describe what the word is
        # Decide whether you want to include these lemmas
        lems = [lem for lem in syn.lemmas()]
        if len(lems) > 0:
            print(lems, ": ", syn.definition())
            print("Do you want to include these into the food log? (y if yes)")
            user = input()
            # If 'y' then write into file
            if user == 'y':
                # Save the lemmas into the file
                for lem in lems:
                    if lem not in history:
                        f.write(lem.name() + ",")

                # Decide whether you want to include these hypos
                hypos = syn.hyponyms()
                if len(hypos) > 0:
                    print("hypos: ")

                    # Add all hyponyms for this word to queue
                    for idx in range(len(hypos)):
                        print(idx, ": ", hypos[idx].name(), ": ", hypos[idx].definition())

                    print("Do you want to include these into the food log and queue?")
                    print("Enter the ones you want to include as a comma delimited list: <1, 2, 6> OR enter 'all' for all")
                    user = input()

                    # If all, keep them all
                    if user == "all":
                        for hypo in hypos:
                            if hypo not in history:
                                f.write(hypo.lemmas()[0].name() + ",")
                                queue.append(hypo)
                    # Otherwise, get the indices of the hypos the user specified to keep and keep them
                    ids = user.split(',')
                    for id in ids:
                        if not id.isnumeric() or int(id) > len(hypos):
                            continue
                        hypo = hypos[int(id)]
                        if hypo not in history:
                            f.write(hypo.lemmas()[0].name() + ",")
                            queue.append(hypo)

            # If exit, close the file and stop this func
            if user == "exit!":
                f.close()
                return