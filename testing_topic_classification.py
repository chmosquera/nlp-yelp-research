import nltk, spacy
from nltk.tokenize import sent_tokenize
from sense2vec import Sense2VecComponent
nltk.download('punkt')
# Resources for determining similarity: Spacy, sense2vec
s2v_path = "D:\\Programs\\Python37x64\\nlp_config\\s2v_reddit_2015_md"
spacy_lg_path = 'D:\\Programs\\Python37x64\\nlp_config\\venv\\Lib\\site-packages\\en_core_web_lg\\en_core_web_lg-2.2.5'
nlp = spacy.load(spacy_lg_path)
s2v = Sense2VecComponent(nlp.vocab).from_disk(s2v_path)
nlp.add_pipe(s2v)


seeds = {}
seeds['food']="food drink"
seeds['atms']= "atmosphere place environment"
seeds['serv']= "server management time"
seeds['prce']= "money expensive"


def avg(my_list):
    if len(my_list) == 0:
        return 0
    return sum(my_list) / len(my_list)


def get_noun_toks(doc):
    return [tok for tok in doc if tok.tag_.startswith('N')]


def calculate_similarity(noun, seed):
    if not seed.has_vector or not noun.has_vector:
        if not seed.has_vector:
            print("Warning: No vector forund for '" + seed.text + "' using spacy module.")
        elif not noun.has_vector:
            print("Warning: No vector found for '" + noun.text + "' using spacy module.")
        return 0

    return noun.similarity(seed)

def calculate_similarity_s2v(noun, seed):
    if not seed._.in_s2v or not noun._.in_s2v:
        return calculate_similarity(noun, seed)
    else:
        return seed._.s2v_similarity(noun)                    


#  Input seeds/nouns as a list of spacy tokens
def categorize(sp_seeds, sp_nouns, use_s2v=True):


    # compute each nouns' similarity to each seed using vectors
    noun_scores = {}
    for noun in sp_nouns:
        score_per_seed = []
        for seed in sp_seeds:
            if use_s2v:
                sim_score = calculate_similarity_s2v(noun, seed)
            else:
                sim_score = calculate_similarity(noun,seed)
            print(noun.text, sim_score)
            score_per_seed.append(sim_score)

        # only care about max score from all similarity(seeds) scores
        noun_scores[noun.text] = max(score_per_seed)

    return noun_scores



# Assigns a single category to the given text
def topicClassification(text, seeds={}, category=['food', 'atms', 'serv', 'prce'], use_s2v=True, soft=True):

    # split into sentences
    sentences = sent_tokenize(text.lower())
    score_matrices = []				# score matrix per sentence

    # Calculate a score matrix for each sentence
    # A score matrix contains a similarity score for each [noun] X [category]
    for sentence in sentences:
        # convert sentence to spacy doc, this gives context when using s2v
        doc = nlp(sentence)

        # 1 Get nouns as spacy tokens, needed for s2v
        nouns = get_noun_toks(doc)

        # 2 Perform Seeded-Topic-Classification
        topic_scores = {}
        for cat in category:
            noun_scores = categorize(nlp(seeds[cat]), nouns, use_s2v=use_s2v)
            scores = [score for noun,score in noun_scores.items()]            
            topic_scores[cat] = avg(scores)

        # sort dict of avg scores
        sorted_score_matr = [(k,v) for k,v in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)]
        # print(sentence, sorted_score_matr)		
        # 3 Result: most similar topics
        score_matrices.append(sorted_score_matr)

    # Normalize the set of assigned topics (nearest topic is the assigned one)
    highest_topics = [ts[0] for ts in score_matrices]   # [[topic,score],...]
    topics = [ts[0] for ts in highest_topics]         # [topic,...]
    normalized_topics = normalizeTopics(topics, category=category, soft=soft)
    # print(topics)
    print("normalized", normalized_topics)

    print(text, normalized_topics)
    return normalized_topics


# input: list of most_similar topics per sentence
# output: []
def normalizeTopics(topics, category=['food', 'atms', 'serv', 'prce'], soft=True):
    cnt_topics = {cat:0 for cat in category}
    print("soft: ", soft)
    # soft normalization - 1 if topic exists
    if not soft:
        for topic in topics:
            if topic not in category:
                print("Error: Missing topic from list of categories. Topic='" + topic + '"')
            else:
                cnt_topics[topic] = 1
        return cnt_topics

    else:
        # count all topics through text
        for topic in topics:
            if topic not in category:
                print("Error: Missing topic from list of categories. Topic='" + topic + '"')
            else:
                cnt_topics[topic]+= 1
            # print("cnt_topic: ", cnt_topics)

        # normalize 0-1
        norm_topic_scores = {}
        total = len(topics)  
        for topic,cnt in cnt_topics.items():
            # print("pair: ", topic, score)
            norm_topic_scores[topic] = cnt / total

        return norm_topic_scores